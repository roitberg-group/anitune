from typing import OrderedDict, Tuple, Optional, NamedTuple
import warnings
import torch
import math
from torch import Tensor
from torchani.models import ANI1x, ANI2x
from torchani.aev.cutoffs import _parse_cutoff_fn
from torchani.aev import AEVComputer
from torchani.aev.aev_terms import StandardRadial
from torchani.aev.cutoffs import CutoffDummy
from torchani.nn import SpeciesConverter
from torchani.atomics import standard


class SpeciesChargesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor
    charges: Tensor


class ExpDistanceDecay(torch.nn.Module):
    def __init__(self, prefactor: float = 1.0, factor: float = 1.0):
        super().__init__()
        self.register_parameter('factor', torch.nn.Parameter(torch.tensor(factor)))
        self.register_parameter('prefactor', torch.nn.Parameter(torch.tensor(prefactor)))

    def forward(self, x: Tensor) -> Tensor:
        return (self.prefactor**2) * torch.exp(-(self.factor**2) * x)


class Smoothstep(torch.nn.Module):
    def __init__(self, edges):
        super().__init__()
        self.low = edges[0]
        self.high = edges[1]
        self.delta = self.high - self.low

    def forward(self, x):
        x = torch.clamp((x - self.low) / self.delta, 0, 1)
        return x.pow(3) * (x * (6 * x - 15) + 10)


class ErfDamp(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return torch.erf(self.factor * x)


class DummyDamp(torch.nn.Module):
    def forward(self, x):
        return 1


def dipole_pass(atom: str = 'H', dims=None):
    dim_tuple = (384, 96) if dims is None else dims
    dims_for_atoms = {'H': dim_tuple,
                      'C': dim_tuple,
                      'N': dim_tuple,
                      'O': dim_tuple}
    out = standard(dims_for_atoms[atom], activation=torch.nn.GELU(),
                    bias=False, classifier_out=None)
    out.out_size = dims_for_atoms['H'][-1]
    return out


def dipole_neighbor_pass(atom: str = 'H', dims=None):
    # input must be output of first pass
    dim_tuple = (96, 48) if dims is None else dims
    dims_for_atoms = {'H': dim_tuple,
                      'C': dim_tuple,
                      'N': dim_tuple,
                      'O': dim_tuple}
    out = standard(dims_for_atoms[atom], activation=torch.nn.GELU(),
                    bias=False, classifier_out=None)
    out.out_size = dims_for_atoms['H'][-1]
    return out


def autodipole_simple_transform_pass(atom: str = 'H', dims=None, elements=('H', 'C', 'N', 'O')):
    dim_tuple = (96, 48) if dims is None else dims
    dims_for_atoms = {k: dim_tuple for k in elements}
    out = standard(dims_for_atoms[atom], activation=torch.nn.GELU(),
                    bias=False, classifier_out=None)
    out.out_size = dims_for_atoms['H'][-1]
    return out


def dipole_simple_transform(atom: str = 'H', dims=None):
    dim_tuple = (144, 96) if dims is None else dims
    dims_for_atoms = {'H': dim_tuple,
                      'C': dim_tuple,
                      'N': dim_tuple,
                      'O': dim_tuple}
    out = standard(dims_for_atoms[atom], activation=torch.nn.GELU(),
                    bias=False, classifier_out=1)
    out.out_size = dims_for_atoms['H'][-1]
    return out


def dipole_nonlocal_transform(atom: str = 'H', dims=None):
    dim_tuple = (144, 96) if dims is None else dims
    dims_for_atoms = {'H': dim_tuple,
                      'C': dim_tuple,
                      'N': dim_tuple,
                      'O': dim_tuple}
    out = standard(dims_for_atoms[atom], activation=torch.nn.GELU(),
                    bias=False, classifier_out=2)
    out.out_size = dims_for_atoms['H'][-1]
    return out


class LocalMessagePassing(torch.nn.Module):
    # Module that allows for local communication between different atoms within
    # a given cutoff
    @staticmethod
    def ensureOrderedDict(modules):
        if not isinstance(modules, OrderedDict):
            raise ValueError("An ordered dict is expected")
        return modules

    def __init__(self,
                 module_dicts,
                 neighbor_module_dicts,
                 final_transform,
                 module_out_sizes=None,
                 neighbor_out_sizes=None,
                 cutoff_fn="smooth",
                 cutoff=5.2,
                 nonlocal_charge_redistribution: bool = False,
                 regularize: bool = False,
                 decay_fn: torch.nn.Module = None,
                 embed_charge: bool = False,
                 charge_embed_dim: int = 20,
                 embed_at_start: bool = False,
                 embed_start_dim: int = None):
        super().__init__()
        self.nonlocal_charge_redistribution = nonlocal_charge_redistribution
        self.regularize = regularize
        # The modules, neighbor modules and final transform have some restrictions
        # on their sizes.
        # (1) The input size of the first pass must be the AEV size
        # (2) the input size of the first neighbor pass must be the output of the
        #     first pass
        # (3) The input size of the next pass module must be out_size_pass_0 + out_size_neigh_0
        #     and it is convenient if this is less than in_size_pass_0
        # and so on and so forth
        if module_out_sizes is None:
            module_out_sizes = [next(iter(mdict.values())).out_size for mdict in module_dicts]
        if neighbor_out_sizes is None:
            neighbor_out_sizes = [next(iter(mdict.values())).out_size for mdict in neighbor_module_dicts]

        if embed_charge:
            assert not embed_at_start
            self._charge_embedding = torch.nn.Embedding(num_embeddings=14, embedding_dim=module_out_sizes[0], padding_idx=0)
            self.embed_charge = True
            self.embed_at_start = False
        elif embed_at_start:
            assert not embed_charge
            if embed_start_dim is not None:
                warnings.warn("embed start dim is deprecated, please use charge_embed_dim")
                charge_embed_dim = embed_start_dim
            self._charge_embedding = torch.nn.Embedding(num_embeddings=14, embedding_dim=charge_embed_dim, padding_idx=0)
            self.embed_charge = False
            self.embed_at_start = True
            self.charge_embed_dim = charge_embed_dim
        else:
            self.embed_charge = False
            self.embed_at_start = False

        # The current implementation needs the output sizes to be passed explicitly,
        # or alternative for the modules to have an out_size attribute
        # the different passes are stored in a list of moduledicts
        self._passes = torch.nn.ModuleList([])
        self._neighbor_passes = torch.nn.ModuleList([])
        self._final_transform = torch.nn.ModuleDict(final_transform)
        self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        self.decay_fn = decay_fn if decay_fn is not None else ExpDistanceDecay()
        self.cutoff = cutoff
        self.register_buffer('module_out_sizes', torch.tensor(module_out_sizes, dtype=torch.long))
        self.register_buffer('neighbor_out_sizes', torch.tensor(neighbor_out_sizes, dtype=torch.long))
        # module dicts must be an iterable of ordered dictionaries of modules
        for dict_ in module_dicts:
            self._passes.append(torch.nn.ModuleDict(self.ensureOrderedDict(dict_)))
        for dict_ in neighbor_module_dicts:
            self._neighbor_passes.append(torch.nn.ModuleDict(self.ensureOrderedDict(dict_)))

    @torch.jit.export
    def forward(self,
                species_aev: Tuple[Tensor, Tensor],
                atom_index12: Tensor,
                distances: Tensor,
                total_charges: Optional[Tensor] = None) -> Tensor:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        # Obtain the atomic energies associated with a given tensor of AEV's
        species_ = species.view(-1)
        aev = aev.view(-1, aev.shape[-1])
        num_atoms = species.shape[1]

        # initial features are the aev
        features = aev
        if self.embed_at_start:
            assert not (total_charges < -6).any(), "less than -6 charge not supported"
            embedding = total_charges + 7
            embedding = embedding.unsqueeze(1).masked_fill(species == -1, 0).long()
            embedding = self._charge_embedding(embedding)
            embedding = embedding.view(-1, self.charge_embed_dim)
            features = torch.cat((features, embedding), dim=-1)
        for j, (module_pass, neighbor_pass) in enumerate(zip(self._passes, self._neighbor_passes)):
            # The pass for each atomic number gets accumulated into an
            # "internal features" tensor
            module_out_size = self.module_out_sizes[j]
            internal_features = features.new_zeros((species_.shape[0], module_out_size))
            for i, m in enumerate(module_pass.values()):
                selected_species_idx = (species_ == i).nonzero().view(-1)
                if selected_species_idx.shape[0] > 0:
                    input_ = features.index_select(0, selected_species_idx)
                    internal_features.index_add_(0, selected_species_idx, m(input_))
            if self.embed_charge and j == 0:
                assert not (total_charges < -6).any(), "less than -6 charge not supported"
                embedding = total_charges + 7
                embedding = embedding.unsqueeze(1).masked_fill(species == -1, 0).long()
                embedding = self._charge_embedding(embedding)
                embedding = embedding.view(-1, module_out_size)
                internal_features *= embedding
            # At this point output has a shape A', G
            # Where A' is padded atoms and G is the AEV dimension
            #
            # I want first to duplicate this and then
            # I'll add in each position a weighted sum of the aev's of all pairs of
            # each atom.
            neighbor_out_size = self.neighbor_out_sizes[j]
            neighbor_features = internal_features.new_zeros(size=(internal_features.shape[0], neighbor_out_size))
            neighbor_merged_features = neighbor_features.detach().clone()

            # The neighbor representations are the same as the internal ones
            # but passed through a simple linear transformation, different for each atom type
            for i, m in enumerate(neighbor_pass.values()):
                selected_species_idx = (species_ == i).nonzero().view(-1)
                if selected_species_idx.shape[0] > 0:
                    input_ = internal_features.index_select(0, selected_species_idx)
                    neighbor_features.index_add_(0, selected_species_idx, m(input_))

            # This is a distance decay,
            decay = self.decay_fn(distances) * self.cutoff_fn(distances, self.cutoff)
            # There may be a more efficient way to do this than flipud, but
            # something like it is definitely needed
            neighbor_features = neighbor_features[atom_index12.flipud()] * decay.view(1, -1, 1)
            neighbor_merged_features.index_add_(0, atom_index12.view(-1), neighbor_features.view(-1, neighbor_out_size))
            features = torch.cat((internal_features, neighbor_merged_features), dim=-1)
        # once we are done with the loop we get some final features
        # the final features go through a linear transformation to output the charges
        precharges = features.new_zeros(size=(species_.shape[0],))
        if self.nonlocal_charge_redistribution:
            distribution_factors = features.new_zeros(size=(species_.shape[0],))
        for i, m in enumerate(self._final_transform.values()):
            selected_species_idx = (species_ == i).nonzero().view(-1)
            if selected_species_idx.shape[0] > 0:
                input_ = features.index_select(0, selected_species_idx)
                # I'm not sure why view(-1) is needed here...
                m_out = m(input_)
                if self.nonlocal_charge_redistribution:
                    # I possibly want here to predict two values!, a charge, and a value that will
                    # redistribute the charge over all other atoms
                    precharges.index_add_(0, selected_species_idx, m_out[:, 0])
                    distribution_factors.index_add_(0, selected_species_idx, m_out[:, 1])
                else:
                    precharges.index_add_(0, selected_species_idx, m_out.view(-1))

        precharges = precharges.view(-1, num_atoms)
        if self.nonlocal_charge_redistribution:
            distribution_factors = distribution_factors.view(-1, num_atoms)
        # precharges has shape (C, A')
        if total_charges is not None:
            total_precharges = precharges.sum(-1)
            dummy_mask = (species != -1)
            if self.nonlocal_charge_redistribution:
                distribution_factors = torch.exp(-distribution_factors**2) + 1e-12  # epsilon for regulatization of division
                if self.regularize:
                    distribution_factors += 1e-5
                distribution_factors = distribution_factors.masked_fill(~dummy_mask, 0.0)
                factors = (distribution_factors / distribution_factors.sum(dim=-1, keepdim=True))
            else:
                factors = 1 / dummy_mask.sum(-1, keepdim=True)
            charges = precharges + (total_charges - total_precharges).unsqueeze(-1) * factors
            # Charges of dummy atoms are manually set to zero
            charges.masked_fill_(~dummy_mask, 0.0)
        else:
            charges = precharges
        return species, charges, precharges


class SimpleLocalDipoleMP(torch.nn.Module):
    def __init__(self, elements, use_cuda_extension=True, main_dims=None,
                 neighbor_dims=None, final_dims=None, embed_charge=False,
                 embed_at_start=False, embed_start_dim=20,
                 nonlocal_charge_redistribution=False, charge_embed_dim: int = 14):
        super().__init__()
        self._aev_computer = AEVComputer.like_1x(use_cuda_extension=use_cuda_extension)
        self._converter = SpeciesConverter(elements)
        module_dicts = [OrderedDict([(e, dipole_pass(e, main_dims)) for e in elements])]
        neighbor_module_dicts = [OrderedDict([(e, dipole_neighbor_pass(e, neighbor_dims)) for e in elements])]
        if nonlocal_charge_redistribution:
            simple_transform = OrderedDict([(e, dipole_nonlocal_transform(e, final_dims)) for e in elements])
        else:
            simple_transform = OrderedDict([(e, dipole_simple_transform(e, final_dims)) for e in elements])

        self._mpnn = LocalMessagePassing(module_dicts, neighbor_module_dicts, simple_transform,
                                         lembed_charge=embed_charge, embed_at_start=embed_at_start,
                                         nonlocal_charge_redistribution=nonlocal_charge_redistribution,
                                         charge_embed_dim=charge_embed_dim)

    def forward(self, species_coordinates, total_charges, cell=None, pbc=None):
        species, coordinates = self._converter(species_coordinates)
        atom_index12, _, diff_vectors, distances = self._aev_computer.neighborlist(species, coordinates, cell, pbc)
        aev = self._aev_computer._compute_aev(species, atom_index12, diff_vectors, distances)
        species, charges, precharges = self._mpnn((species, aev), atom_index12, distances, total_charges)
        return species, None, charges


class DipolesAndEnergies1x(torch.nn.Module):
    def __init__(self,
                 energy_model_kwargs,
                 main_dims=None,
                 neighbor_dims=None,
                 final_dims=None,
                 atomic_maker=None,
                 simple_erf_damp=False,
                 smoothstep_damp=False,
                 smoothstep_edges=(0.0, 5.2),
                 embed_charge=False,
                 embed_at_start: bool = False,
                 embed_start_dim: int = 20,
                 embed_and_couple_start: bool = False,
                 embed_and_couple_dim_start: int = None,
                 coulomb_terms: bool = False,
                 coulomb_eta=None,
                 coulomb_shifts=None,
                 charge_embed_dim: int = 20,
                 detach_charges_for_energy: bool = False,
                 nonlocal_charge_redistribution: bool = False,
                 regularize: bool = False,
                 charge_loss: bool = False,
                 use_2x: bool = False,
                 detach_charges: bool = False):
        super().__init__()
        self.charge_loss = charge_loss
        self.detach_charges = detach_charges
        self.detach_charges_for_energy = detach_charges_for_energy
        if use_2x:
            model_class = ANI2x
        else:
            model_class = ANI1x
        if atomic_maker is not None:
            energies_model = model_class(**energy_model_kwargs, atomic_maker=atomic_maker)
        else:
            energies_model = model_class(**energy_model_kwargs)
        self.use_coulomb_aev = coulomb_terms
        if coulomb_terms:
            coulomb_eta = torch.tensor([3.5]) if coulomb_eta is None else torch.tensor(coulomb_eta)
            # this adds 32 terms to a 4-species AEV
            coulomb_shifts = torch.linspace(-2., 2., 8) if coulomb_shifts is None else torch.tensor(coulomb_shifts)
            self.coulomb_terms = StandardRadial(coulomb_eta, coulomb_shifts, cutoff=math.inf, cutoff_fn=CutoffDummy())
        self.num_species = energies_model.aev_computer.num_species
        self.use_cuda_extension = energies_model.aev_computer.use_cuda_extension
        elements = energies_model.get_chemical_symbols()
        self.elements = elements

        module_dicts = [OrderedDict([(e, dipole_pass(e, main_dims)) for e in elements])]
        neighbor_module_dicts = [OrderedDict([(e, dipole_neighbor_pass(e, neighbor_dims)) for e in elements])]
        if nonlocal_charge_redistribution:
            simple_transform = OrderedDict([(e, dipole_nonlocal_transform(e, final_dims)) for e in elements])
        else:
            simple_transform = OrderedDict([(e, dipole_simple_transform(e, final_dims)) for e in elements])
        self._mpnn = LocalMessagePassing(module_dicts, neighbor_module_dicts, simple_transform, embed_charge=embed_charge,
                                         embed_at_start=embed_at_start,
                                         nonlocal_charge_redistribution=nonlocal_charge_redistribution,
                                         regularize=regularize,
                                         charge_embed_dim=charge_embed_dim)
        self.simple_erf_damp = simple_erf_damp
        self.aev_computer = energies_model.aev_computer
        self.neural_networks = energies_model.neural_networks
        self.species_converter = energies_model.species_converter
        self.energy_shifter = energies_model.energy_shifter
        self.neighborlist = self.aev_computer.neighborlist
        self.neighborlist.cutoff = math.inf
        self.aev_cutoff = self.aev_computer.radial_terms.cutoff
        if simple_erf_damp:
            assert not smoothstep_damp
            self.damp_fn = ErfDamp(0.4)
        elif smoothstep_damp:
            assert not simple_erf_damp
            # Some good possibilites
            # 0.0, 5.2
            # 1.6, 5.2
            # 1.6, 2.8
            self.damp_fn = Smoothstep(smoothstep_edges)
        else:
            self.damp_fn = DummyDamp()

        if embed_and_couple_start:
            assert not embed_at_start
            assert not embed_charge
            if embed_and_couple_dim_start:
                warnings.warn("embed_and_couple_dim_start is deprecated please use charge_embed_dim")
                charge_embed_dim = embed_and_couple_dim_start
            self._charge_embedding = torch.nn.Embedding(num_embeddings=14, embedding_dim=charge_embed_dim, padding_idx=0)
            self.embed_and_couple_start = True
        else:
            self.embed_and_couple_start = False

    def get_chemical_symbols(self):
        return self.elements

    def forward(self, species_coordinates, total_charges, cell=None, pbc=None):
        species, coordinates = self.species_converter(species_coordinates)
        full_pairs, _, full_diff_vectors, full_distances = self.neighborlist(species,
                                                                             coordinates,
                                                                             cell,
                                                                             pbc)
        atom_index12, _, diff_vectors, distances = self.neighborlist._rescreen_with_cutoff(self.aev_cutoff,
                                                                                           full_pairs,
                                                                                           full_diff_vectors,
                                                                                           full_distances)
        aev_indices = (full_distances <= self.aev_cutoff).nonzero().flatten()

        if self.use_cuda_extension:
            if not self.aev_computer.cuaev_is_initialized:
                self.aev_computer._init_cuaev_computer()
                self.aev_computer.cuaev_is_initialized = True
            assert pbc is None or (not pbc.any()), "cuaev currently does not support PBC"
            aev = self.aev_computer._compute_cuaev(species, coordinates)
        else:
            aev = self.aev_computer._compute_aev(species, atom_index12, diff_vectors, distances)
        if self.embed_and_couple_start:
            assert not (total_charges < -6).any(), "less than -6 charge not supported"
            embedding = total_charges + 7
            embedding = embedding.unsqueeze(1).masked_fill(species == -1, 0).long()
            embedding = self._charge_embedding(embedding)
            aev = torch.cat((aev, embedding), dim=-1)
        species, charges, precharges = self._mpnn((species, aev), atom_index12, distances, total_charges)
        pair_product_charges = charges.view(-1)[full_pairs].prod(0)
        if self.detach_charges:
            pair_product_charges = pair_product_charges.detach()
        coulomb_terms = pair_product_charges * (1 / full_distances)
        coulomb_terms *= self.damp_fn(full_distances)

        # here I will grab only the distances within the AEV cutoff, and screen the coulomb
        # terms this way
        if self.use_coulomb_aev:
            if self.detach_charges_for_energy:
                local_coulombic_energies = pair_product_charges.detach() * (1 / full_distances) * self.damp_fn(full_distances)
                local_coulombic_energies = local_coulombic_energies.index_select(0, aev_indices)
            else:
                local_coulombic_energies = coulomb_terms.index_select(0, aev_indices)
            coulomb_aev = self._compute_coulomb_aev(species.shape[0],
                                                    species.shape[1],
                                                    species.flatten()[atom_index12],
                                                    local_coulombic_energies,
                                                    atom_index12)
            aev = torch.cat((aev, coulomb_aev), dim=-1)

        short_range_energies = self.neural_networks((species, aev)).energies

        molecule_indices = torch.div(full_pairs[0], species.shape[1], rounding_mode='floor')
        short_range_energies.index_add_(0, molecule_indices, coulomb_terms)
        species, energies = self.energy_shifter((species, short_range_energies))
        if self.charge_loss:
            return species, energies, charges, precharges
        return SpeciesChargesEnergies(species, energies, charges)

    def _compute_coulomb_aev(self, num_molecules: int, num_atoms: int,
                             species12: Tensor, energies: Tensor,
                             atom_index12: Tensor) -> Tensor:

        coulomb_terms_ = self.coulomb_terms(energies)
        coulomb_aev = coulomb_terms_.new_zeros(
            (num_molecules * num_atoms * self.num_species,
             self.coulomb_terms.sublength))
        index12 = atom_index12 * self.num_species + species12.flip(0)
        coulomb_aev.index_add_(0, index12[0], coulomb_terms_)
        coulomb_aev.index_add_(0, index12[1], coulomb_terms_)
        coulomb_aev = coulomb_aev.reshape(num_molecules, num_atoms, self.coulomb_terms.sublength * self.num_species)
        return coulomb_aev
