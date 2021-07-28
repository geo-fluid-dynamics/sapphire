from dataclasses import dataclass, field


@dataclass
class EutecticBinaryAlloy:
    """Material data class"""

    melting_temperature_of_solvent: float

    eutectic_temperature: float

    eutectic_concentration: float


@dataclass
class Materials:
    """Materials library data class"""
    sodium_chloride_dissolved_in_water: EutecticBinaryAlloy = field(init=False)

    def __post_init__(self):

        self.sodium_chloride_dissolved_in_water = EutecticBinaryAlloy(
            melting_temperature_of_solvent=0.,  # [deg C]
            eutectic_temperature=-21.,  # [deg C]
            eutectic_concentration=23.,  # [% wt. NaCl]
            )


MATERIALS = Materials()
