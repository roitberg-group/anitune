# type: ignore
from typing import OrderedDict, Optional, NamedTuple
import torch
import math
from torch import Tensor
from torchani.nn import SpeciesConverter
from torchani.aev.cutoffs import _parse_cutoff_fn
from torchani.aev.aev_terms import StandardRadial
from torchani.aev.cutoffs import CutoffDummy
from torchani.aev import AEVComputer
from torchani.utils import EnergyShifter, sorted_gsaes
from torchani.atomics import standard


class SpeciesChargesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor
    charges: Tensor


class ExpDistanceDecay(torch.nn.Module):
    def __init__(self, prefactor: float = 1.0, factor: float = 1.0):
        super().__init__()
        self.register_parameter("factor", torch.nn.Parameter(torch.tensor(factor)))
        self.register_parameter(
            "prefactor", torch.nn.Parameter(torch.tensor(prefactor))
        )

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


def MLP(atom: str = "H", dims=None, elements=("H", "C", "N", "O"), classifier_out=None):
    # final dim_tuple = (144, 96, 48) if dims is None else dims
    # neighbor dim_tuple = (144, 96, 48) if dims is None else dims
    dim_tuple = (384, 96) if dims is None else dims
    dims_for_atoms = {k: dim_tuple for k in elements}
    out = standard(
        dims_for_atoms[atom],
        activation=torch.nn.GELU(),
        bias=False,
        classifier_out=classifier_out,
    )
    out.out_size = dims_for_atoms["H"][-1] if classifier_out is None else classifier_out
    return out


class LocalMessagePassing(torch.nn.Module):
    # Module that allows for local communication between different atoms within
    # a given cutoff
    @staticmethod
    def ensureOrderedDict(modules):
        if not isinstance(modules, OrderedDict):
            raise ValueError("An ordered dict is expected")
        return modules

    def __init__(
        self,
        module_dicts,
        neighbor_module_dicts,
        final_transform,
        cutoff_fn="smooth",
        cutoff=5.2,
        learned_redistribution: bool = False,
        decay_fn: torch.nn.Module = None,
    ):
        super().__init__()
        self.learned_redistribution = learned_redistribution
        # The modules, neighbor modules and final transform have some restrictions
        # on their sizes.
        # (1) The input size of the first pass must be the AEV size
        # (2) the input size of the first neighbor pass must be the output of the
        #     first pass
        # (3) The input size of the next pass module must be out_size_pass_0 + out_size_neigh_0
        #     and it is convenient if this is less than in_size_pass_0
        # and so on and so forth
        module_out_sizes = [
            next(iter(mdict.values())).out_size for mdict in module_dicts
        ]
        if len(neighbor_module_dicts[0]) != 0:
            neighbor_out_sizes = [
                next(iter(mdict.values())).out_size for mdict in neighbor_module_dicts
            ]
        else:
            neighbor_out_sizes = [0]

        # The current implementation needs the output sizes to be passed explicitly,
        # or alternative for the modules to have an out_size attribute
        # the different passes are stored in a list of moduledicts
        self._passes = torch.nn.ModuleList([])
        self._neighbor_passes = torch.nn.ModuleList([])
        self._final_transform = torch.nn.ModuleDict(final_transform)
        self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        self.decay_fn = decay_fn if decay_fn is not None else ExpDistanceDecay()
        self.cutoff = cutoff
        self.register_buffer(
            "module_out_sizes", torch.tensor(module_out_sizes, dtype=torch.long)
        )
        self.register_buffer(
            "neighbor_out_sizes", torch.tensor(neighbor_out_sizes, dtype=torch.long)
        )
        # module dicts must be an iterable of ordered dictionaries of modules
        for dict_ in module_dicts:
            self._passes.append(torch.nn.ModuleDict(self.ensureOrderedDict(dict_)))
        for dict_ in neighbor_module_dicts:
            self._neighbor_passes.append(
                torch.nn.ModuleDict(self.ensureOrderedDict(dict_))
            )
        if len(neighbor_module_dicts) == 0:
            self._neighbor_passes = [None]
        if len(self._final_transform) == 0:
            self._final_transform = None

    @torch.jit.export
    def forward(
        self,
        species: Tensor,
        in_features: Tensor,
        atom_index12: Tensor = None,
        distances: Tensor = None,
        total_charges: Optional[Tensor] = None,
    ) -> Tensor:
        assert species.shape == in_features.shape[:-1]
        # Obtain the atomic energies associated with a given tensor of AEV's
        species_ = species.view(-1)
        in_features = in_features.view(-1, in_features.shape[-1])
        num_atoms = species.shape[1]
        features = in_features
        for j, (module_pass, neighbor_pass) in enumerate(
            zip(self._passes, self._neighbor_passes)
        ):
            # The pass for each atomic number gets accumulated into an
            # "internal features" tensor
            module_out_size = self.module_out_sizes[j]
            internal_features = features.new_zeros((species_.shape[0], module_out_size))
            for i, m in enumerate(module_pass.values()):
                selected_species_idx = (species_ == i).nonzero().view(-1)
                if selected_species_idx.shape[0] > 0:
                    input_ = features.index_select(0, selected_species_idx)
                    internal_features.index_add_(0, selected_species_idx, m(input_))
            # At this point output has a shape A', G
            # Where A' is padded atoms and G is the AEV dimension
            #
            # I want first to duplicate this and then
            # I'll add in each position a weighted sum of the aev's of all pairs of
            # each atom.

            # The neighbor representations are the same as the internal ones
            # but passed through a simple linear transformation, different for each atom type
            if len(neighbor_pass) > 0:
                neighbor_out_size = self.neighbor_out_sizes[j]
                neighbor_features = internal_features.new_zeros(
                    size=(internal_features.shape[0], neighbor_out_size)
                )
                neighbor_merged_features = neighbor_features.detach().clone()
                for i, m in enumerate(neighbor_pass.values()):
                    selected_species_idx = (species_ == i).nonzero().view(-1)
                    if selected_species_idx.shape[0] > 0:
                        input_ = internal_features.index_select(0, selected_species_idx)
                        neighbor_features.index_add_(0, selected_species_idx, m(input_))

                # This is a distance decay,
                decay = self.decay_fn(distances) * self.cutoff_fn(
                    distances, self.cutoff
                )
                # There may be a more efficient way to do this than flipud, but
                # something like it is definitely needed
                neighbor_features = neighbor_features[
                    atom_index12.flipud()
                ] * decay.view(1, -1, 1)
                neighbor_merged_features.index_add_(
                    0,
                    atom_index12.view(-1),
                    neighbor_features.view(-1, neighbor_out_size),
                )
                features = torch.cat(
                    (internal_features, neighbor_merged_features), dim=-1
                )
            else:
                assert distances is None
                assert atom_index12 is None
                features = internal_features
        # once we are done with the loop we get some final features
        # the final features go through a linear transformation to output the charges
        precharges = features.new_zeros(size=(species_.shape[0],))
        if self.learned_redistribution:
            distribution_factors = features.new_zeros(size=(species_.shape[0],))

        if self._final_transform is not None:
            for i, m in enumerate(self._final_transform.values()):
                selected_species_idx = (species_ == i).nonzero().view(-1)
                if selected_species_idx.shape[0] > 0:
                    input_ = features.index_select(0, selected_species_idx)
                    # I'm not sure why view(-1) is needed here...
                    m_out = m(input_)
                    if self.learned_redistribution:
                        # I possibly want here to predict two values!, a charge, and a value that will
                        # redistribute the charge over all other atoms
                        precharges.index_add_(0, selected_species_idx, m_out[:, 0])
                        distribution_factors.index_add_(
                            0, selected_species_idx, m_out[:, 1]
                        )
                    else:
                        precharges.index_add_(0, selected_species_idx, m_out.view(-1))
        else:
            precharges = features
        precharges = precharges.view(-1, num_atoms)
        if self.learned_redistribution:
            distribution_factors = distribution_factors.view(-1, num_atoms)

        # precharges has shape (C, A')
        if total_charges is not None:
            total_precharges = precharges.sum(-1)
            dummy_mask = species != -1
            if self.learned_redistribution:
                distribution_factors = torch.exp(-(distribution_factors**2)) + 1.0e-5
                distribution_factors = distribution_factors.masked_fill(
                    ~dummy_mask, 0.0
                )
                factors = distribution_factors / distribution_factors.sum(
                    dim=-1, keepdim=True
                )
            else:
                factors = 1 / dummy_mask.sum(-1, keepdim=True)
            charges = (
                precharges + (total_charges - total_precharges).unsqueeze(-1) * factors
            )
            # Charges of dummy atoms are manually set to zero
            charges.masked_fill_(~dummy_mask, 0.0)
        else:
            charges = precharges
        return species, charges, precharges


class DipoleEnergyLean(torch.nn.Module):
    def __init__(
        self,
        elements,
        aev_computer=None,
        aev_computer_kwargs=None,
        like_1x: bool = False,
        like_2x: bool = False,
        damp_fn="erf",
        damp_fn_kwargs=None,
        embed_charge: bool = True,
        embed_charge_dim: int = 20,
        embed_charge_num: int = 7,
        learned_redistribution: bool = False,
        coulomb_eta=None,
        coulomb_shifts=None,
        detach_charges: bool = False,
        functional="B973c",
        basis_set="def2mtzvp",
        energy_decays=("soft", None, None),
        charge_decays=("soft", "hard", "soft"),
        charge_loss: bool = False,
    ):
        super().__init__()
        aev_computer = AEVComputer if aev_computer is None else aev_computer
        aev_computer_kwargs = (
            dict() if aev_computer_kwargs is None else aev_computer_kwargs
        )
        if energy_decays[1] is None:
            main_energy_decay = self._calc_decay(
                energy_decays[0], (3, 4, 4), (4, 8, 10), (4, 9, 10)
            )
            neighbors_energy_decay = tuple()
            final_energy_decay = tuple()
        else:
            main_energy_decay = self._calc_decay(energy_decays[0], (3, 4), (4, 6))
            neighbors_energy_decay = self._calc_decay(energy_decays[1], (2, 3), (2, 4))
            final_energy_decay = self._calc_decay(energy_decays[2], (2,), (4,))

        if charge_decays[1] is None:
            main_charge_decay = self._calc_decay(charge_decays[0], (2, 3, 4), (3, 3, 6))
            neighbors_charge_decay = tuple()
            final_charge_decay = tuple()
        else:
            main_charge_decay = self._calc_decay(
                charge_decays[0], (2, 3), (5, 6), (6, 8)
            )
            neighbors_charge_decay = self._calc_decay(charge_decays[1], (2, 3), (3, 3))
            final_charge_decay = self._calc_decay(charge_decays[2], (1,), (2,))

        damp_fn_kwargs = dict() if damp_fn_kwargs is None else damp_fn_kwargs
        self.charge_loss = charge_loss
        self.detach_charges = detach_charges

        coulomb_eta = (
            torch.tensor([3.5]) if coulomb_eta is None else torch.tensor(coulomb_eta)
        )
        coulomb_shifts = (
            torch.linspace(-2.0, 2.0, 8)
            if coulomb_shifts is None
            else torch.tensor(coulomb_shifts)
        )

        self.coulomb_terms = StandardRadial(
            coulomb_eta, coulomb_shifts, cutoff=math.inf, cutoff_fn=CutoffDummy()
        )

        if like_1x:
            self.aev_computer = aev_computer.like_1x(**aev_computer_kwargs)
        elif like_2x:
            self.aev_computer = aev_computer.like_2x(**aev_computer_kwargs)
        else:
            self.aev_computer = aev_computer(**aev_computer_kwargs)

        self.num_species = self.aev_computer.num_species
        self.use_cuda_extension = self.aev_computer.use_cuda_extension
        self.elements = elements
        in_features = self.aev_computer.aev_length

        if embed_charge:
            self._charge_embedding = torch.nn.Embedding(
                num_embeddings=2 * embed_charge_num,
                embedding_dim=embed_charge_dim,
                padding_idx=0,
            )
            in_features += embed_charge_dim
            self.embed_charge = True
        else:
            self.embed_charge = False

        if charge_decays[1] is not None:
            main_module_dims = tuple(in_features // j for j in main_charge_decay)
            neighbor_dims = tuple(
                main_module_dims[-1] // j for j in neighbors_charge_decay
            )
            final_dims = tuple(
                neighbor_dims[-1] // j + main_module_dims[-1] // j
                for j in final_charge_decay
            )
            charge_module_dicts = [
                OrderedDict(
                    [
                        (e, MLP(e, elements=elements, dims=main_module_dims))
                        for e in elements
                    ]
                )
            ]
            charge_neighbor_module_dicts = [
                OrderedDict(
                    [
                        (e, MLP(e, elements=elements, dims=neighbor_dims))
                        for e in elements
                    ]
                )
            ]
            charge_final_transform = OrderedDict(
                [
                    (
                        e,
                        MLP(
                            e,
                            dims=final_dims,
                            elements=elements,
                            classifier_out=2 if learned_redistribution else 1,
                        ),
                    )
                    for e in elements
                ]
            )
        else:
            main_module_dims = tuple(in_features // j for j in main_charge_decay)
            charge_module_dicts = [
                OrderedDict(
                    [
                        (
                            e,
                            MLP(
                                e,
                                dims=main_module_dims,
                                elements=elements,
                                classifier_out=2 if learned_redistribution else 1,
                            ),
                        )
                        for e in elements
                    ]
                )
            ]
            charge_neighbor_module_dicts = [OrderedDict()]
            charge_final_transform = OrderedDict()
        self._charge_mpnn = LocalMessagePassing(
            charge_module_dicts,
            charge_neighbor_module_dicts,
            charge_final_transform,
            learned_redistribution=learned_redistribution,
        )

        in_features += self.num_species * self.coulomb_terms.sublength
        if energy_decays[1] is not None:
            main_module_dims = tuple(in_features // j for j in main_energy_decay)
            neighbor_dims = tuple(
                main_module_dims[-1] // j for j in neighbors_energy_decay
            )
            final_dims = tuple(
                neighbor_dims[-1] // j + main_module_dims[-1] // j
                for j in final_energy_decay
            )

            energy_module_dicts = [
                OrderedDict(
                    [
                        (e, MLP(e, elements=elements, dims=main_module_dims))
                        for e in elements
                    ]
                )
            ]
            energy_neighbor_module_dicts = [
                OrderedDict(
                    [
                        (e, MLP(e, elements=elements, dims=neighbor_dims))
                        for e in elements
                    ]
                )
            ]
            energy_final_transform = OrderedDict(
                [(e, MLP(e, elements=elements, dims=final_dims)) for e in elements]
            )
        else:
            main_module_dims = tuple(in_features // j for j in main_energy_decay)
            energy_module_dicts = [
                OrderedDict(
                    [
                        (
                            e,
                            MLP(
                                e,
                                elements=elements,
                                dims=main_module_dims,
                                classifier_out=1,
                            ),
                        )
                        for e in elements
                    ]
                )
            ]
            energy_neighbor_module_dicts = [OrderedDict()]
            energy_final_transform = OrderedDict()
        self._energy_mpnn = LocalMessagePassing(
            energy_module_dicts, energy_neighbor_module_dicts, energy_final_transform
        )

        self.species_converter = SpeciesConverter(elements)
        self.energy_shifter = EnergyShifter(
            sorted_gsaes(elements, functional=functional, basis_set=basis_set)
        )
        self.neighborlist = self.aev_computer.neighborlist
        self.neighborlist.cutoff = math.inf
        self.aev_cutoff = self.aev_computer.radial_terms.cutoff
        if damp_fn == "smoothstep":
            damp_args = dict()
            damp_args.update(damp_fn_kwargs)
            self.damp_fn = Smoothstep(**damp_args)
        elif damp_fn == "erf":
            damp_args = {"factor": 0.4}
            damp_args.update(damp_fn_kwargs)
            self.damp_fn = ErfDamp(**damp_args)
        else:
            self.damp_fn = DummyDamp()

    def _calc_decay(self, decay, soft, hard, veryhard=object()):
        decay_ = (1,)
        if decay == "soft":
            decay_ += soft
        elif decay == "hard":
            decay_ += hard
        elif decay == "veryhard":
            decay_ += veryhard
        elif decay is None:
            return None
        else:
            decay_ += decay
        return decay_

    def get_chemical_symbols(self):
        return self.elements

    def forward(self, species_coordinates, total_charges, cell=None, pbc=None):
        species, coordinates = self.species_converter(species_coordinates)
        full_pairs, _, full_diff_vectors, full_distances = self.neighborlist(
            species, coordinates, cell, pbc
        )
        atom_index12, _, diff_vectors, distances = (
            self.neighborlist._rescreen_with_cutoff(
                self.aev_cutoff, full_pairs, full_diff_vectors, full_distances
            )
        )
        aev_indices = (full_distances <= self.aev_cutoff).nonzero().flatten()
        if self.use_cuda_extension:
            if not self.aev_computer.cuaev_is_initialized:
                self.aev_computer._init_cuaev_computer()
                self.aev_computer.cuaev_is_initialized = True
            assert pbc is None or (
                not pbc.any()
            ), "cuaev currently does not support PBC"
            aev = self.aev_computer._compute_cuaev(species, coordinates)
        else:
            aev = self.aev_computer._compute_aev(
                species, atom_index12, diff_vectors, distances
            )
        if self.embed_charge:
            assert not (total_charges < -6).any(), "less than -6 charge not supported"
            embedding = total_charges + 7
            embedding = embedding.unsqueeze(1).masked_fill(species == -1, 0).long()
            embedding = self._charge_embedding(embedding)
            aev = torch.cat((aev, embedding), dim=-1)
        species, charges, precharges = self._charge_mpnn(
            species, aev, atom_index12, distances, total_charges
        )
        pair_product_charges = charges.view(-1)[full_pairs].prod(0)
        coulomb_terms = pair_product_charges * (1 / full_distances)
        coulomb_terms *= self.damp_fn(full_distances)
        # here I will grab only the distances within the AEV cutoff, and screen the coulomb
        # terms this way
        if self.detach_charges:
            local_coulombic_energies = (
                pair_product_charges.detach()
                * (1 / full_distances)
                * self.damp_fn(full_distances)
            )
            local_coulombic_energies = local_coulombic_energies.index_select(
                0, aev_indices
            )
        else:
            local_coulombic_energies = coulomb_terms.index_select(0, aev_indices)
        coulomb_aev = self._compute_coulomb_aev(
            species.shape[0],
            species.shape[1],
            species.flatten()[atom_index12],
            local_coulombic_energies,
            atom_index12,
        )
        aev = torch.cat((aev, coulomb_aev), dim=-1)
        a, short_range_energies, c = self._energy_mpnn(species, aev)
        short_range_energies = short_range_energies.sum(-1)  # sum over atoms
        molecule_indices = torch.div(
            full_pairs[0], species.shape[1], rounding_mode="floor"
        )
        short_range_energies.index_add_(0, molecule_indices, coulomb_terms)
        species, energies = self.energy_shifter((species, short_range_energies))
        if self.charge_loss:
            return species, energies, charges, precharges
        return SpeciesChargesEnergies(species, energies, charges)

    def _compute_coulomb_aev(
        self,
        num_molecules: int,
        num_atoms: int,
        species12: Tensor,
        energies: Tensor,
        atom_index12: Tensor,
    ) -> Tensor:
        coulomb_terms_ = self.coulomb_terms(energies)
        coulomb_aev = coulomb_terms_.new_zeros(
            (num_molecules * num_atoms * self.num_species, self.coulomb_terms.sublength)
        )
        index12 = atom_index12 * self.num_species + species12.flip(0)
        coulomb_aev.index_add_(0, index12[0], coulomb_terms_)
        coulomb_aev.index_add_(0, index12[1], coulomb_terms_)
        coulomb_aev = coulomb_aev.reshape(
            num_molecules, num_atoms, self.coulomb_terms.sublength * self.num_species
        )
        return coulomb_aev
