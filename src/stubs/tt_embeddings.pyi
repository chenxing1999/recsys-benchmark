from typing import Any, List, Optional, Tuple, overload

import torch

@overload
def cache_backward_dense(
    arg0: int,
    arg1: torch.Tensor,
    arg2: torch.Tensor,
    arg3: torch.Tensor,
    arg4: float,
    arg5: torch.Tensor,
) -> torch.Tensor: ...
@overload
def cache_backward_dense() -> Any: ...
@overload
def cache_backward_rowwise_adagrad_approx(
    arg0: int,
    arg1: torch.Tensor,
    arg2: torch.Tensor,
    arg3: torch.Tensor,
    arg4: float,
    arg5: float,
    arg6: torch.Tensor,
    arg7: torch.Tensor,
) -> None: ...
@overload
def cache_backward_rowwise_adagrad_approx() -> Any: ...
@overload
def cache_backward_sgd(
    arg0: int,
    arg1: torch.Tensor,
    arg2: torch.Tensor,
    arg3: torch.Tensor,
    arg4: float,
    arg5: torch.Tensor,
) -> None: ...
@overload
def cache_backward_sgd() -> Any: ...
@overload
def cache_forward(
    arg0: int,
    arg1: int,
    arg2: torch.Tensor,
    arg3: torch.Tensor,
    arg4: torch.Tensor,
    arg5: torch.Tensor,
) -> None: ...
@overload
def cache_forward() -> Any: ...
@overload
def cache_populate(
    arg0: int,
    arg1: List[int],
    arg2: List[int],
    arg3: List[int],
    arg4: List[torch.Tensor],
    arg5: torch.Tensor,
    arg6: torch.Tensor,
    arg7: torch.Tensor,
    arg8: torch.Tensor,
    arg9: torch.Tensor,
) -> None: ...
@overload
def cache_populate() -> Any: ...
def preprocess_indices_sync(
    arg0: torch.Tensor,
    arg1: torch.Tensor,
    arg2: int,
    arg3: bool,
    arg4: torch.Tensor,
    arg5: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]: ...
@overload
def tt_adagrad_backward(
    arg0: int,
    arg1: int,
    arg2: float,
    arg3: float,
    arg4: List[int],
    arg5: List[int],
    arg6: List[int],
    arg7: torch.Tensor,
    arg8: int,
    arg9: torch.Tensor,
    arg10: torch.Tensor,
    arg11: torch.Tensor,
    arg12: torch.Tensor,
    arg13: List[torch.Tensor],
    arg14: List[torch.Tensor],
) -> None: ...
@overload
def tt_adagrad_backward() -> Any: ...
@overload
def tt_dense_backward(
    arg0: int,
    arg1: int,
    arg2: List[int],
    arg3: List[int],
    arg4: List[int],
    arg5: torch.Tensor,
    arg6: int,
    arg7: torch.Tensor,
    arg8: torch.Tensor,
    arg9: torch.Tensor,
    arg10: torch.Tensor,
    arg11: List[torch.Tensor],
) -> List[torch.Tensor]: ...
@overload
def tt_dense_backward() -> Any: ...
@overload
def tt_forward(
    arg0: int,
    arg1: int,
    arg2: int,
    arg3: int,
    arg4: List[int],
    arg5: List[int],
    arg6: List[int],
    arg7: torch.Tensor,
    arg8: int,
    arg9: torch.Tensor,
    arg10: torch.Tensor,
    arg11: torch.Tensor,
    arg12: List[torch.Tensor],
) -> torch.Tensor: ...
@overload
def tt_forward() -> Any: ...
@overload
def tt_sgd_backward(
    arg0: int,
    arg1: int,
    arg2: float,
    arg3: List[int],
    arg4: List[int],
    arg5: List[int],
    arg6: torch.Tensor,
    arg7: int,
    arg8: torch.Tensor,
    arg9: torch.Tensor,
    arg10: torch.Tensor,
    arg11: torch.Tensor,
    arg12: List[torch.Tensor],
) -> None: ...
@overload
def tt_sgd_backward() -> Any: ...
@overload
def update_cache_state(
    arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor
) -> None: ...
@overload
def update_cache_state() -> Any: ...
