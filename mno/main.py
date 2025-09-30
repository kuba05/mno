from mno.linesearch import ternary_search
from mno.function import Function
from mno.stopping_condition import SmallChangeCondition
from mno.my_types import array_to_vec


def main() -> None:
    """Run the whole thing."""
    print(
        ternary_search(
            Function(lambda a: (a[0] + 2) ** 2 - a[1] ** 2, dim_in=2, dim_out=1),
            stopping_condition=SmallChangeCondition(),
            point_a=array_to_vec([0, 0]),
            point_b=array_to_vec([-3, 0]),
        )
    )


if __name__ == "__main__":
    main()
