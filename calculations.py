import math

def calculate_Ka(phi):
    """
    Oblicza współczynnik parcia czynnego K_a na podstawie kąta tarcia wewnętrznego phi.
    Parametry:
    phi (float): kąt tarcia wewnętrznego w stopniach
    Zwraca:
    float: współczynnik K_a
    """
    phi_rad = math.radians(phi)
    Ka = math.tan(math.pi / 4 - phi_rad / 2) ** 2
    return Ka

def calculate_Ka_coulomb(phi, delta, beta, theta):
    """
    Oblicza współczynnik parcia czynnego K_a według wzoru Coulomba.
    phi: Kąt tarcia wewnętrznego gruntu [stopnie]
    delta: Kąt tarcia gruntu o ścianę [stopnie]
    beta: Kąt nachylenia ściany od pionu [stopnie] (dla ściany pionowej beta = 0)
    theta: Kąt nachylenia terenu [stopnie] (dla terenu poziomego theta = 0)
    """
    phi_rad = math.radians(phi)
    delta_rad = math.radians(delta)
    beta_rad = math.radians(beta)
    theta_rad = math.radians(theta)
    
    numerator = math.sin(phi_rad + delta_rad) * math.sin(phi_rad - beta_rad - theta_rad)
    denominator = math.cos(delta_rad + theta_rad) * math.cos(beta_rad)
    Ka = (numerator / denominator)
    Ka = Ka / (1 + math.sqrt((math.sin(phi_rad + delta_rad) * math.sin(phi_rad - beta_rad - theta_rad)) / (math.cos(delta_rad + theta_rad) * math.cos(beta_rad))))
    Ka = Ka ** 2
    return Ka
   

def calculate_Ka_terzaghi(phi):
    """
    Oblicza współczynnik parcia czynnego K_a według metody Terzaghiego.
    W tej metodzie K_a może być mniejsze niż wartość Rankine'a dla gruntów spoistych.
    """
    phi_rad = math.radians(phi)
    Ka = (1 - math.sin(phi_rad)) / (1 + math.sin(phi_rad))  # Wzór Rankine'a
    
    # Terzaghi wprowadza korekty dla gruntów spoistych
    # Możesz tu dodać własne korekty lub tabele
    return Ka  # Tymczasowo zwracamy K_a z Rankine'a

def calculate_Ka_tschebotarioff(phi):
    """
    Oblicza współczynnik parcia czynnego K_a według metody Tschebotarioffa.
    """
    phi_rad = math.radians(phi)
    Ka = (1 - math.sin(phi_rad)) / (2 * (1 + math.sin(phi_rad)))
    return Ka


def calculate_settlement(e0, delta_sigma, H0, Cc, sigma_p0):
    """
    Oblicza osiadanie gruntu spowodowane odwodnieniem.
    e0: Początkowy współczynnik porowatości
    delta_sigma: Zmiana efektywnego naprężenia pionowego [kPa]
    H0: Grubość warstwy podlegającej konsolidacji [m]
    Cc: Współczynnik konsolidacji
    sigma_p0: Początkowe efektywne naprężenie pionowe [kPa]
    """
    if sigma_p0 == 0:
        sigma_p0 = 1  # Uniknięcie dzielenia przez zero, ustawiamy minimalną wartość
    settlement = (Cc * H0) / (1 + e0) * math.log10((sigma_p0 + delta_sigma) / sigma_p0)
    return settlement





def calculate_soil_pressure(Ka, gamma, z, c=0):
    """
    Oblicza jednostkowe parcie gruntu na głębokości z.
    Parametry:
    Ka (float): współczynnik parcia czynnego
    gamma (float): ciężar objętościowy gruntu [kN/m^3]
    z (float): głębokość [m]
    c (float): spójność gruntu [kPa] (domyślnie 0)
    Zwraca:
    float: jednostkowe parcie gruntu [kPa]
    """
    sigma_h = Ka * gamma * z - 2 * c * math.sqrt(Ka)
    return sigma_h

def calculate_soil_pressure_with_water(Ka, gamma, gamma_sat, gamma_w, z, zw, c=0):
    """
    Oblicza jednostkowe parcie gruntu na głębokości z, uwzględniając poziom wody gruntowej.
    Parametry:
    Ka (float): współczynnik parcia czynnego
    gamma (float): ciężar objętościowy gruntu [kN/m³]
    gamma_sat (float): ciężar objętościowy nasycony gruntu [kN/m³]
    gamma_w (float): ciężar objętościowy wody [kN/m³]
    z (float): głębokość [m]
    zw (float): poziom wody gruntowej [m]
    c (float): spójność gruntu [kPa] (domyślnie 0)
    Zwraca:
    float: jednostkowe parcie gruntu [kPa]
    """
    if z <= zw:
        # Powyżej poziomu wody gruntowej
        sigma_h = Ka * gamma * z - 2 * c * math.sqrt(Ka)
    else:
        # Poniżej poziomu wody gruntowej
        sigma_h = Ka * (gamma * zw + (gamma_sat - gamma_w) * (z - zw)) - 2 * c * math.sqrt(Ka)
    return sigma_h


def calculate_total_soil_pressure_with_surcharge_and_water(
    H_i, gamma_i, gamma_sat_i, phi_i, c_i, q, d, zw, gamma_w, metoda_parcia, delta=0, beta=0, theta=0
):
    sigma_h_total = 0
    z = 0
    for i in range(len(H_i)):
        phi = phi_i[i]
        c = c_i[i]
        gamma = gamma_i[i]
        gamma_sat = gamma_sat_i[i]
        H = H_i[i]
        z_start = z
        z_end = z + H
        z_mid = (z_start + z_end) / 2

         # Wybór metody obliczeń
        if metoda_parcia == 'coulomb':
            Ka = calculate_Ka_coulomb(phi, delta, beta, theta)
        elif metoda_parcia == 'terzaghi':
            Ka = calculate_Ka_terzaghi(phi)
        elif metoda_parcia == 'tschebotarioff':
            Ka = calculate_Ka_tschebotarioff(phi)
        else:
            Ka = calculate_Ka(phi)  # Domyślnie Rankine

        # Obliczanie współczynnika redukcyjnego
        reduction = reduction_factor(d, z_mid)

        # Obliczanie parcia gruntu z uwzględnieniem wody gruntowej
        sigma_h = calculate_soil_pressure_with_water(
            Ka, gamma, gamma_sat, gamma_w, z_mid, zw, c
        )

        # Dodanie parcia od obciążenia zewnętrznego z uwzględnieniem redukcji
        sigma_h += calculate_surcharge_pressure(Ka, q, reduction)

        sigma_h_total += sigma_h * H
        z = z_end
    return sigma_h_total




def calculate_max_moment(sigma_h_total, H, liczba_kotwien):
    """
    Oblicza maksymalny moment zginający na blacie szalunkowym.
    """
    if liczba_kotwien == 0:
        # Ściana wolno stojąca
        M_max = (sigma_h_total * H) / 6
    elif liczba_kotwien == 1:
        # Ściana z jedną kotwą
        M_max = (sigma_h_total * H) / 8
    elif liczba_kotwien == 2:
        # Ściana z dwiema kotwami
        M_max = (sigma_h_total * H) / 10
    else:
        # Dla większej liczby kotwień, moment maleje
        M_max = (sigma_h_total * H) / (6 + liczba_kotwien * 2)
    return M_max



def check_bending_strength(M_max, W, f_y, gamma_M=1.0):
    """
    Sprawdza warunek nośności na zginanie.
    """
    sigma = M_max / W  # kN·m / m³ = kN/m² = kPa
    f_y_kPa = f_y * 1000  # Konwersja MPa do kPa
    return sigma <= f_y_kPa / gamma_M

def calculate_section_modulus(b, t):
    """
    Oblicza wskaźnik wytrzymałości przekroju W dla przekroju prostokątnego.
    Parametry:
    b (float): to szerokość przekroju (w poziomie) [m]
    t (float): to wysokość przekroju (w pionie) [m]
    Zwraca:
    float: wskaźnik wytrzymałości przekroju W [m³]
    """
    W = (b * t**2) / 6
    return W
def calculate_surcharge_pressure(Ka, q, reduction):
    """
    Oblicza dodatkowe parcie gruntu wynikające z obciążenia zewnętrznego, uwzględniając współczynnik redukcyjny.
    """
    return Ka * q * reduction

def reduction_factor(d, z):
    """
    Oblicza współczynnik redukcyjny w zależności od odległości d i głębokości z.
    Parametry:
    d (float): odległość od krawędzi wykopu [m]
    z (float): głębokość [m]
    Zwraca:
    float: współczynnik redukcyjny (wartość między 0 a 1)
    """
    n = d / z if z != 0 else 0
    if n >= 2:
        return 0  # Brak wpływu obciążenia
    elif n > 0:
        return 1 - n / 2
    else:
        return 1  # Pełny wpływ obciążenia

