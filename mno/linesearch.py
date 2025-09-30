from mno.function import Function
from mno.stopping_condition import StoppingCondition
from mno.my_types import Vec, float_to_vec


def ternary_search(
    function: Function,
    stopping_condition: StoppingCondition,
    point_a: Vec,
    point_b: Vec,
) -> Vec:
    """Ternary search on a general function on a line between point_a and point_b."""
    direction = point_b - point_a

    def helper(k: Vec) -> Vec:
        point = k[0] * direction + point_a
        return function(point)

    line_function = Function(function=helper, dim_in=1, dim_out=function.get_dim()[1])

    left = float_to_vec(0)
    right = float_to_vec(1)
    prev_mid_point = None
    while stopping_condition(
        function=function,
        cur_point=(left + right) / 2,
        prev_point=prev_mid_point,
    ):
        prev_mid_point = (left + right) / 2
        left_mid = (left + 2 * right) / 3
        right_mid = (2 * left + right) / 3
        if line_function(left_mid) > line_function(right_mid):
            left = left_mid
        else:
            right = right_mid

    return (left + right) / 2
