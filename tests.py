import flux_calculations

f = flux_calculations.calc_flux(1, 20, 100000, 0.5)
assert(abs(2.0514932183173233e-05 - f)<1e-6)

print 'ok'
