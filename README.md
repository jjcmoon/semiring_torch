# semiring_torch

Run your pytorch code on any semiring with a single line of code!
Semiring_torch is built on top of [autoray](https://github.com/jcmgray/autoray).

> Warning: this is a proof of concept. Expect bugs and missing features.

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
result = x1[:, :, None] + x2[None, :, :]
result = torch.logsumexp(result, dim=1)
result = result.exp()
```
</td>
<td>

```python
from autoray import numpy as torch
from semiring_torch import logarithmic_semiring

with logarithmic_semiring:
    x1 = torch.tensor([[0.1, 0.6], [0.1, 0.4]])
    x2 = torch.tensor([[0.5, 0.3], [0.2, 0.1]])
    result = x1 @ x2
```

</td>
</tr>
</table>

### Supported semirings
Currently only the logarithmic semiring is supported, but more semirings can be added easily.