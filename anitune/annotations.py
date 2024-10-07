r"""
Type aliases, common annotations used throughout the ANITune code
"""

import typing as tp

Scalar = tp.Union[bool, int, float, str, None]
ScalarTuple = tp.Tuple[str, tp.Union[bool, int, float, str, None]]
