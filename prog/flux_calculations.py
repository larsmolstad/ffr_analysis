def calc_flux(slope, T, p=101000, chamber_height=0.5):
    """ 
    Calculates the flux based on the slope of the ppm-vs-time curve

    slope: ppm/s
    T: Celcius
    p: Pa (pressure)
    chamber_height: m
    flux: mol/m2/s 
    """
    R = 8.314
    Tk = T + 273.15
    mol_gas_in_chamber_per_area = p * chamber_height / (R * Tk)
    return slope * 1e-6 * mol_gas_in_chamber_per_area
