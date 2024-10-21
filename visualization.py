import matplotlib.pyplot as plt

def plot_soil_pressure(z_values, sigma_h_values):
    plt.figure()
    plt.plot(sigma_h_values, z_values)
    plt.gca().invert_yaxis()
    plt.xlabel("Parcie gruntu σ_h [kPa]")
    plt.ylabel("Głębokość z [m]")
    plt.title("Rozkład parcia gruntu")
    plt.grid(True)
    plt.show()
