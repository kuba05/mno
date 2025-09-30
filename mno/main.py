from mno.function import Function
from mno.linesearch import TernarySearch
from mno.my_types import array_to_vec
from mno.stopping_condition import SmallChangeCondition


def main() -> None:
    """Run the whole thing."""
    print(
        TernarySearch()
        .set_function(
            Function(lambda a: (a[0] + 2) ** 2 - a[1] ** 2, dim_in=2, dim_out=1)
        )
        .set_stopping_condition(stopping_condition=SmallChangeCondition())
        .set_interval(
            point_a=array_to_vec([0, 0]),
            point_b=array_to_vec([-3, 0]),
        )
    )


if __name__ == "__main__":
    main()
