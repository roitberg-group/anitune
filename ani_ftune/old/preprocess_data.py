
# type: ignore
from pathlib import Path

import torch

import torchani
from torchani.utils import tqdm
from torchani import datasets
from torchani.repulsion import StandaloneRepulsionCalculator
from torchani.dispersion import StandaloneDispersionD3
from torchani.short_range_basis import StandaloneEnergySRB
from torchani.aev.cutoffs import CutoffSmooth

if __name__ == "__main__":
    # from torchani.short_range_basis import StandaloneEnergySRB
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elements = torchani.models.ANI2x().get_chemical_symbols()
    # Code to perform dataset preprocessing
    proc_ds = datasets.ANIDataset(
        "./BondBreaking-reduced-B973c-def2mTZVP_proc.h5",
        create=True,
        grouping="by_num_atoms",
    )
    # ds = datasets.ANI2x(download=False, basis_set='def2mTZVP', functional='B973c', skip_integrity_check=True)
    ds = datasets.ANIDataset("./BondBreaking-reduced-B973c-def2mTZVP.h5")
    rep = StandaloneRepulsionCalculator(
        cutoff=5.1,
        cutoff_fn=CutoffSmooth(order=2),
        elements=elements,
        periodic_table_index=True,
    ).to(device)
    disp = StandaloneDispersionD3(
        cutoff=8.0,
        cutoff_fn=CutoffSmooth(order=2),
        elements=elements,
        periodic_table_index=True,
        functional="B97-3c",
    ).to(device)
    srb = StandaloneEnergySRB(
        cutoff=5.1,
        cutoff_fn=CutoffSmooth(order=2),
        elements=elements,
        periodic_table_index=True,
    ).to(device)
    try:
        max_size = 2500
        for name, j, group in tqdm(
            ds.chunked_numpy_items(max_size=2500, chem_symbols=False),
            total=ds.num_conformers / max_size,
        ):
            extra_keys = {
                "D3-two-body.energy-corrections": [],
                "xTB-repulsion.energy-corrections": [],
                "D3-two-body.force-corrections": [],
                "xTB-repulsion.force-corrections": [],
                "SRB.energy-corrections": [],
                "SRB.force-corrections": [],
            }
            coordinates = torch.from_numpy(group["coordinates"]).float().to(device)
            species = torch.from_numpy(group["species"]).long().to(device)
            if species.shape[0] == 0:
                continue

            coordinates.requires_grad_(True)
            repulsion_energy = rep((species, coordinates)).energies
            repulsion_force = -torch.autograd.grad(repulsion_energy.sum(), coordinates)[
                0
            ]

            coordinates = coordinates.detach().requires_grad_(True)
            dispersion_energy = disp((species, coordinates)).energies
            dispersion_force = -torch.autograd.grad(
                dispersion_energy.sum(), coordinates
            )[0]

            coordinates = coordinates.detach().requires_grad_(True)
            srb_energy = srb((species, coordinates)).energies
            srb_force = -torch.autograd.grad(srb_energy.sum(), coordinates)[0]

            extra_keys["D3-two-body.energy-corrections"].append(
                dispersion_energy.detach().cpu()
            )
            extra_keys["D3-two-body.force-corrections"].append(
                dispersion_force.detach().cpu()
            )
            extra_keys["xTB-repulsion.energy-corrections"].append(
                repulsion_energy.detach().cpu()
            )
            extra_keys["xTB-repulsion.force-corrections"].append(
                repulsion_force.detach().cpu()
            )
            extra_keys["SRB.force-corrections"].append(srb_force.detach().cpu())
            extra_keys["SRB.energy-corrections"].append(srb_energy.detach().cpu())

            extra_keys = {k: torch.cat(v).numpy() for k, v in extra_keys.items()}
            extra_keys["processed_energies"] = (
                group["energies"]
                - extra_keys["D3-two-body.energy-corrections"]
                - extra_keys["xTB-repulsion.energy-corrections"]
            )
            extra_keys["processed_forces"] = (
                group["forces"]
                - extra_keys["D3-two-body.force-corrections"]
                - extra_keys["xTB-repulsion.force-corrections"]
            )
            group.update(extra_keys)
            proc_ds.append_conformers(name.split("/")[-1], group)
        proc_ds.repack()
    except Exception as e:
        Path("./2x_wB973c_processed_good_cuts.h5").resolve().unlink()
        raise e

    meta = ds.metadata[next(iter(ds.metadata.keys()))]
    meta["README"] += (
        "\nThis dataset was preprocessed for faster simulations."
        "\nprocessed_energies = energies - D3-two-body.energy-corrections - xTB-repulsion.energy-corrections"
        "\nprocessed_forces = forces - D3-two-body.force-corrections - xTB-repulsion.force-corrections"
        "\nThe D3-two-body and xTB-repulsion terms were calculated using the repulsion and dispersion ANI modules,"
        "\nSRB contributions are included in the dataset but are not subtracted when processing energies or forces"
        "\nAdequate cutoffs were used in this case (5.1 for rep and srb, and 8.0 for disp)"
        "\nThis dataset contains molecules in situations close to bond-breaking"
    )
    proc_ds.set_metadata({"0": meta})
