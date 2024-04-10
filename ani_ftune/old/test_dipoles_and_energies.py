
# type: ignore
from dipole_model import DipolesAndEnergies1x
import torch

device = "cpu"
# .. note:: The coordinates are in Angstrom, and the energies you get are in Hartree
coordinates = torch.tensor(
    [
        [
            [0.03192167, 0.00638559, 0.01301679],
            [-0.83140486, 0.39370209, -0.26395324],
            [-0.66518241, -0.84461308, 0.20759389],
            [0.45554739, 0.54289633, 0.81170881],
            [0.66091919, -0.16799635, -0.91037834],
        ]
    ],
    requires_grad=True,
    device=device,
)
# In periodic table, C = 6 and H = 1
species = torch.tensor([[6, 1, 1, 1, 1]], device=device)

m = DipolesAndEnergies1x(
    energy_model_kwargs={"periodic_table_index": True, "use_cuda_extension": True}
)
total_charges = torch.tensor([0])

species, charges, energies = m((species, coordinates), total_charges)
forces = -torch.autograd.grad(energies.sum(), coordinates)[0]
