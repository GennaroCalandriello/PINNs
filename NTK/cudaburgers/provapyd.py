import burgers2d
import numpy as np
burgers2d.setup2d()

bc = np.array(burgers2d.get_left_bc())
print("Boundary condition at x=0:", bc)
print("len", bc.shape, "type", type(bc))