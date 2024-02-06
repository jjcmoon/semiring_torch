# semiring_torch

Run your pytorch code on any semiring with a single line of code!
Semiring_torch is built on top of [autoray](https://github.com/jcmgray/autoray).

### Example

By using the logarithmic semiring, you can easily write numerically stable code. 
In the following example, we compute a matrix product in log-space.

<table>
<tr>
<th>Regular torch</th>
<th>semiring_torch</th>
</tr>
<tr>
<td>

```python
import torch

x1 = torch.tensor([[0.1, 0.6], [0.1, 0.4]])
x2 = torch.tensor([[0.5, 0.3], [0.2, 0.1]])
x1 = x1.log()
x2 = x2.log()
result2 = x1[:, :, None] + x2[None, :, :]
result2 = torch.logsumexp(result2, dim=1)
result2 = result2.exp()
```
</td>
<td>

```python
import autoray as ar
from autoray import numpy as np
from semiring_torch import *

with ar.backend_like('log_torch'):
    x1 = np.array([[0.1, 0.6], [0.1, 0.4]])
    x2 = np.array([[0.5, 0.3], [0.2, 0.1]])
    result = x1 @ x2
```

</td>
</tr>
</table>

### Supported semirings
Currently only the logarithmic semiring is supported. More semirings will be added soon.