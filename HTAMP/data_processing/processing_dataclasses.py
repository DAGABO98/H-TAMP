
from dataclasses import dataclass

@dataclass
class HospitalDataFiles:
    medications_orders: str
    blood_pressure_orders: str
    heart_rate_orders: str
    respiratory_rate_orders: str
    temperature_orders: str
    oxygen_saturation_orders: str
    visits_data: str