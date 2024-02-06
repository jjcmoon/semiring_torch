# semiring_torch

> Run your pytorch code on any semiring with a single line of code!


### Example

By using the logarithmic semiring, you can easily write numerically stable code. 
In the following example, we compute the dot product of two vectors.

<table>
<tr>
<th>Regular torch</th>
<th>semiring_torch</th>
</tr>
<tr>
<td>

```python
import torch




x1 = torch.tensor([0.01, 0.06]).log()
x2 = torch.tensor([0.03, 0.04]).log()
dot_product = torch.logsumexp(x1 + x2, dim=0).exp()

```
</td>
<td>

```python
import autoray as ar
from autoray import numpy as np
from semiring_torch import *

with ar.backend_like('log_semiring'):
    x1 = np.array([0.01, 0.06])
    x2 = np.array([0.03, 0.04])
    dot_product = np.dot(x1, x2)

```

</td>
</tr>
</table>

### Supported semirings
Currently only the logarithmic semiring is supported. More semirings will be added soon.