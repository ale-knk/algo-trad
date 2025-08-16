from typing import Dict, List, Optional, Union


class IndicatorName:
    """
    Class that encapsulates the structure of an indicator name, including its base name,
    parameters, whether it uses returns, and any component names.

    This provides a more structured and OOP approach to handle indicator naming.
    """

    def __init__(
        self,
        base_name: str,
        parameters: Optional[
            Union[List[Union[int, float]], Dict[str, Union[int, float]]]
        ] = None,
        uses_returns: bool = False,
        component: Optional[str] = None,
    ):
        """
        Initialize an IndicatorName.

        Args:
            base_name: The base name of the indicator (e.g., "MA", "RSI", "MACD")
            parameters: Parameters for the indicator (e.g., periods, deviations)
            uses_returns: Whether this indicator uses returns data instead of prices
            component: For complex indicators, the component name (e.g., "macd_line", "upper_band")
        """
        self.base_name = base_name
        self.parameters = parameters if parameters is not None else []
        self.uses_returns = uses_returns
        self.component = component

    @classmethod
    def from_string(cls, name_str: str) -> "IndicatorName":
        """
        Parse an indicator name string into an IndicatorName object.

        Args:
            name_str: String representation of an indicator name
                     (e.g., "MA_14", "MACD_12_26_9-returns-macd_line")

        Returns:
            IndicatorName: Parsed indicator name object
        """
        parts = name_str.split("-")
        base_with_params = parts[0]  # e.g., "MA_14" or "MACD_12_26_9"

        # Determine if it uses returns
        uses_returns = len(parts) > 1 and parts[1] == "returns"

        # Determine component (if any)
        component = None
        if len(parts) > 1 and parts[1] != "returns":
            component = parts[1]
        elif len(parts) > 2:
            component = parts[2]

        # Parse base name and parameters
        base_parts = base_with_params.split("_")
        base_name = base_parts[0]

        # Extract parameters (convert to appropriate type)
        parameters = []
        for param in base_parts[1:]:
            try:
                # Try to convert to int first
                param_value = int(param)
            except ValueError:
                try:
                    # If not an int, try float
                    param_value = float(param)
                except ValueError:
                    # If not numeric, keep as string
                    param_value = param
            parameters.append(param_value)

        return cls(base_name, parameters, uses_returns, component)

    def to_string(self) -> str:
        """
        Convert the IndicatorName to its string representation.

        Returns:
            str: String representation of the indicator name
        """
        # Start with base name
        name_parts = [self.base_name]

        # Add parameters
        if isinstance(self.parameters, list):
            for param in self.parameters:
                name_parts.append(str(param))
        elif isinstance(self.parameters, dict):
            for key, value in self.parameters.items():
                name_parts.append(f"{key}={value}")

        # Join base and parameters with underscores
        base_with_params = "_".join(name_parts)

        # Add returns suffix if needed
        if self.uses_returns and not self.component:
            return f"{base_with_params}-returns"
        elif self.uses_returns and self.component:
            return f"{base_with_params}-returns-{self.component}"
        elif self.component:
            return f"{base_with_params}-{self.component}"
        else:
            return base_with_params

    def __str__(self) -> str:
        """String representation for printing."""
        return self.to_string()

    def __repr__(self) -> str:
        """Formal representation for debugging."""
        return f"IndicatorName(base_name={self.base_name!r}, parameters={self.parameters!r}, uses_returns={self.uses_returns!r}, component={self.component!r})"

    def __eq__(self, other) -> bool:
        """
        Compare two IndicatorName objects for equality.

        Args:
            other: Another IndicatorName or string to compare with

        Returns:
            bool: True if the indicator names are equal
        """
        if isinstance(other, str):
            other = IndicatorName.from_string(other)

        if not isinstance(other, IndicatorName):
            return False

        return (
            self.base_name == other.base_name
            and self.parameters == other.parameters
            and self.uses_returns == other.uses_returns
            and self.component == other.component
        )

    def __hash__(self) -> int:
        """
        Generate a hash for this IndicatorName.

        Returns:
            int: Hash value
        """
        return hash(
            (
                self.base_name,
                tuple(self.parameters)
                if isinstance(self.parameters, list)
                else frozenset(self.parameters.items()),
                self.uses_returns,
                self.component,
            )
        )

    @property
    def is_complex(self) -> bool:
        """
        Check if this is a complex indicator (with components).

        Returns:
            bool: True if the indicator has components
        """
        return self.component is not None

    def with_component(self, component: str) -> "IndicatorName":
        """
        Create a new IndicatorName with the specified component.

        Args:
            component: Component name

        Returns:
            IndicatorName: New indicator name with the specified component
        """
        return IndicatorName(
            self.base_name, self.parameters, self.uses_returns, component
        )

    def without_component(self) -> "IndicatorName":
        """
        Create a new IndicatorName without any component.

        Returns:
            IndicatorName: New indicator name without component
        """
        return IndicatorName(self.base_name, self.parameters, self.uses_returns, None)
