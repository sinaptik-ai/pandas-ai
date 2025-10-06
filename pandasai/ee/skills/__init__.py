import inspect
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, PrivateAttr


class SkillType(BaseModel):
    """Skill that takes a function usable by pandasai"""

    func: Callable[..., Any]
    description: Optional[str] = None
    name: Optional[str] = None
    _signature: Optional[str] = PrivateAttr()

    def __init__(
        self,
        func: Callable[..., Any],
        description: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the skill.

        Args:
            func: The function from which to create a skill
            description: The description of the skill.
                Defaults to the function docstring.
            name: The name of the function. Mandatory when `func` is a lambda.
                Defaults to the function's name.
            **kwargs: additional params
        """

        name = name or func.__name__
        description = description or func.__doc__
        if description is None:
            # if description is None then the function doesn't have a docstring
            # and the user didn't provide any description
            raise ValueError(
                f"Function must have a docstring if no description is provided for skill {name}."
            )
        signature = f"def {name}{inspect.signature(func)}:"

        super(SkillType, self).__init__(
            func=func, description=description, name=name, **kwargs
        )
        self._signature = signature

    def __call__(self, *args, **kwargs) -> Any:
        """Calls the skill function"""
        return self.func(*args, **kwargs)

    @classmethod
    def from_function(cls, func: Callable, **kwargs: Any) -> "SkillType":
        """
        Creates a skill object from a function

        Args:
            func: The function from which to create a skill

        Returns:
            the `Skill` object

        """
        return cls(func=func, **kwargs)

    def stringify(self):
        return inspect.getsource(self.func)

    def __str__(self):
        return (
            f'<function>\n{self._signature}\n    """{self.description}"""\n</function>'
        )


def skill(*args: Union[str, Callable]) -> Callable:
    """Decorator to create a skill out of functions and automatically add it to the global skills manager.
    Can be used without arguments. The function must have a docstring.

    Args:
        *args: The arguments to the skill

    Examples:
        .. code-block:: python

            @skill
            def compute_flight_prices(offers: pd.DataFrame) -> List[float]:
                \"\"\"Computes the flight prices\"\"\"
                return

            @skill("custom_name")
            def compute_flight_prices(offers: pd.Dataframe) -> List[float]:
                \"\"\"Computes the flight prices\"\"\"
                return
    """

    def _make_skill_with_name(skill_name: str) -> Callable:
        def _make_skill(skill_fn: Callable) -> SkillType:
            skill_obj = SkillType(
                name=skill_name,  # func.__name__ if None
                # when this decorator is used, the function MUST have a docstring
                description=skill_fn.__doc__,
                func=skill_fn,
            )

            # Automatically add the skill to the global skills manager
            try:
                from pandasai.ee.skills.manager import SkillsManager

                SkillsManager.add_skills(skill_obj)
            except ImportError:
                # If SkillsManager is not available, just return the skill
                pass

            return skill_obj

        return _make_skill

    if len(args) == 1 and isinstance(args[0], str):
        # Example: @skill("skillName")
        return _make_skill_with_name(args[0])
    elif len(args) == 1 and callable(args[0]):
        # Example: @skill
        return _make_skill_with_name(args[0].__name__)(args[0])
    elif not args:
        # Covers the case in which a function is decorated with "@skill()"
        # with the intended behavior of "@skill"
        def _func_wrapper(fn: Callable) -> SkillType:
            return _make_skill_with_name(fn.__name__)(fn)

        return _func_wrapper
    else:
        raise ValueError(
            f"Too many arguments for skill decorator. Received: {len(args)}"
        )


__all__ = ["skill", "SkillType"]
