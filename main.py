from dotenv import load_dotenv
load_dotenv()

import os
import math
import io
import json
import base64
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, send_file, render_template, request, redirect, url_for, flash
from calculations import (
    calculate_Ka, calculate_Ka_coulomb, calculate_Ka_terzaghi,
    calculate_Ka_tschebotarioff, calculate_settlement,
    calculate_soil_pressure_with_water,
    calculate_total_soil_pressure_with_surcharge_and_water,
    calculate_max_moment, check_bending_strength, calculate_section_modulus,
    calculate_surcharge_pressure, reduction_factor)

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
from weasyprint import HTML, CSS
from itertools import zip_longest

import plotly.graph_objs as go
import plotly

# Inicjalizacja aplikacji Flask
app = Flask(__name__)
app.config["SECRET_KEY"] = "PRB123"  # Zmień na silny, losowy klucz

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)

# Ładowanie listy haseł z zmiennej środowiskowej
CORRECT_PASSWORDS = os.getenv("REPORT_PASSWORDS", "default_password").split(",")
# Usunięcie nadmiarowych spacji w hasłach
CORRECT_PASSWORDS = [pwd.strip() for pwd in CORRECT_PASSWORDS]

logging.debug(f"CORRECT_PASSWORDS loaded: {CORRECT_PASSWORDS}")

# Definicja ścieżki do folderu z czcionkami
# Zakładając, że folder 'fonts' znajduje się w głównym katalogu projektu
FONT_DIR = os.path.join(os.path.dirname(__file__), 'fonts')  # Jeśli fonts są w 'myapponline/fonts/'
logging.debug(f"FONT_DIR: {FONT_DIR}")
logging.debug(f"DejaVuSans.ttf exists: {os.path.isfile(os.path.join(FONT_DIR, 'DejaVuSans.ttf'))}")
logging.debug(f"DejaVuSans-Bold.ttf exists: {os.path.isfile(os.path.join(FONT_DIR, 'DejaVuSans-Bold.ttf'))}")



# Rejestracja czcionek
try:
    pdfmetrics.registerFont(TTFont('DejaVuSans', os.path.join(FONT_DIR, 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', os.path.join(FONT_DIR, 'DejaVuSans-Bold.ttf')))
    logging.debug("Czcionki zarejestrowane pomyślnie.")
except Exception as e:
    logging.error(f"Błąd podczas rejestracji czcionek: {e}")




# Definicja funkcji pomocniczych
def get_phi_at_depth(z, H_i, phi_values):
    """
    Zwraca wartość kąta tarcia wewnętrznego φ dla danej głębokości z.
    """
    z_accum = 0
    for i, H_layer in enumerate(H_i):
        z_accum += H_layer
        if z <= z_accum:
            return phi_values[i]
    return phi_values[-1]  # Jeśli z jest większe niż całkowita głębokość, zwróć ostatnią wartość

def get_gamma_at_depth(z, H_i, gamma_values, gamma_sat_values, zw):
    """
    Zwraca wartość ciężaru objętościowego γ dla danej głębokości z, uwzględniając poziom wody gruntowej.
    """
    z_accum = 0
    for i, H_layer in enumerate(H_i):
        z_accum += H_layer
        if z <= z_accum:
            if z > zw:
                return gamma_sat_values[i]  # Poniżej poziomu wody gruntowej
            else:
                return gamma_values[i]  # Powyżej poziomu wody gruntowej
    return gamma_values[-1]  # Jeśli z jest większe niż całkowita głębokość, zwróć ostatnią wartość

def get_c_at_depth(z, H_i, c_values):
    """
    Zwraca wartość spójności c dla danej głębokości z.
    """
    z_accum = 0
    for i, H_layer in enumerate(H_i):
        z_accum += H_layer
        if z <= z_accum:
            return c_values[i]
    return c_values[-1]  # Jeśli z jest większe niż całkowita głębokość, zwróć ostatnią wartość

def create_pressure_plot(z_values, sigma_h_values, calculation_method, phi_values, gamma_values, gamma_sat_values, c_values, H_i, zw):
    """
    Tworzy wykres rozkładu parcia gruntu za pomocą Plotly i zwraca go w formie JSON oraz base64.
    """
    print("Tworzenie wykresu interaktywnego...")

    # Debugging: check lengths
    print(f"len(z_values): {len(z_values)}, len(gamma_values): {len(gamma_values)}, len(gamma_sat_values): {len(gamma_sat_values)}")

    # Obliczenie parcia biernego
    sigma_h_passive = []
    for i in range(len(gamma_values)):  # Iteracja po warstwach
        z = z_values[i]
        phi = get_phi_at_depth(z, H_i, phi_values)
        gamma = get_gamma_at_depth(z, H_i, gamma_values, gamma_sat_values, zw)
        c = get_c_at_depth(z, H_i, c_values)
        Kp = (1 + math.sin(math.radians(phi))) / (1 - math.sin(math.radians(phi)))

        # Obliczanie gamma_sub na podstawie głębokości
        if z > zw:
            try:
                gamma_sub = gamma_sat_values[i] - 9.81  # γ' = γ_sat - γ_w
                print(f"Layer {i+1}: z={z}, gamma_sub={gamma_sub}")
            except IndexError:
                print(f"Błąd: gamma_sat_values[{i}] jest poza zakresem.")
                gamma_sub = 0  # Możesz ustawić domyślną wartość lub obsłużyć błąd inaczej
        else:
            try:
                gamma_sub = gamma_values[i]  # Powyżej poziomu wody gruntowej
                print(f"Layer {i+1}: z={z}, gamma_sub={gamma_sub}")
            except IndexError:
                print(f"Błąd: gamma_values[{i}] jest poza zakresem.")
                gamma_sub = 0  # Możesz ustawić domyślną wartość lub obsłużyć błąd inaczej

        sigma_v = gamma_sub * z  # Naprężenie pionowe efektywne
        sigma_h_p = Kp * sigma_v + 2 * c * math.sqrt(Kp)
        sigma_h_passive.append(sigma_h_p)

    # Ustawienie większych rozmiarów figury Plotly
    fig = go.Figure(layout=dict(width=800, height=600))  # Zwiększono szerokość i wysokość

    # Dodajemy parcie czynne
    fig.add_trace(
        go.Scatter(
            x=sigma_h_values,
            y=z_values,
            mode="lines+markers",
            name="Parcie czynne",
            hovertemplate="Parcie czynne: %{x:.2f} kPa<br>" + "Głębokość: %{y:.2f} m<br>" + "<extra></extra>",
        )
    )

    # Dodajemy parcie bierne
    fig.add_trace(
        go.Scatter(
            x=sigma_h_passive,
            y=z_values[:len(gamma_values)],
            mode="lines+markers",
            name="Parcie bierne",
            hovertemplate="Parcie bierne: %{x:.2f} kPa<br>" + "Głębokość: %{y:.2f} m<br>" + "<extra></extra>",
            line=dict(color="green"),
        )
    )

    fig.update_layout(
        title={
            "text": f"Rozkład Parcia Gruntu - Metoda {calculation_method}",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Parcie gruntu σₕ [kPa]",
        yaxis_title="Głębokość z [m]",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        width=800,  # Zwiększona szerokość
        height=600,  # Zwiększona wysokość
    )

    try:
        # Serializacja interaktywnego wykresu do JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        print("Wykres został pomyślnie zserializowany.")
    except Exception as e:
        print(f"Błąd podczas serializacji wykresu: {e}")
        graphJSON = None

    try:
        # Generowanie statycznego obrazu PNG wykresu
        plot_png = fig.to_image(format="png")
        if plot_png:
            # Kodowanie obrazu PNG w base64
            plot_base64 = base64.b64encode(plot_png).decode("utf-8")
            print("Wykres PNG został pomyślnie wygenerowany.")
            print(f"plot_base64 length: {len(plot_base64)}")
        else:
            print("Błąd: plot_png jest pusty.")
            plot_base64 = None
    except Exception as e:
        print(f"Błąd podczas generowania obrazu wykresu: {e}")
        plot_base64 = None

    return graphJSON, plot_base64




def create_cross_section_plot(H_i, zw, H_total, q, d, theta, h_r, metoda_parcia, z_values, sigma_h_values):
    """
    Tworzy przekrój poprzeczny wykopu za pomocą Matplotlib i zwraca go w formie base64.
    """
    from matplotlib.patches import Polygon

    # Utworzenie figury
    fig, ax = plt.subplots(figsize=(6, 8))

    # Rysowanie warstw gruntu po lewej stronie wykopu
    z = 0
    for i, H_layer in enumerate(H_i):
        rect = plt.Rectangle((-1, z), 1, H_layer, facecolor=f"C{i%10}", edgecolor="black")
        ax.add_patch(rect)
        ax.text(-0.5, z + H_layer / 2, f"Warstwa {i+1}", ha="center", va="center", rotation=90, fontsize=10)
        z += H_layer

    # Rysowanie poziomu wody gruntowej
    ax.axhline(y=zw, color="blue", linestyle="--", label="Poziom wody gruntowej")
    ax.text(-1.2, zw, "Poziom wody gruntowej", ha="right", va="center", fontsize=10, color="blue")

    # Rysowanie szalunku na x = 0
    ax.plot([0, 0], [0, z], color="brown", linewidth=5, label="Szalunek")
    ax.text(0.05, z / 2, "Szalunek", ha="left", va="center", rotation=90, fontsize=10, color="brown")

    # Rysowanie rozpory
    rozpora_coords = [
        (0, h_r - 0.05),
        (0, h_r + 0.05),
        (1.0, h_r + 0.05),
        (1.0, h_r - 0.05),
    ]

    rozpora = Polygon(rozpora_coords, closed=True, facecolor="grey", edgecolor="black")
    ax.add_patch(rozpora)
    ax.text(0.5, h_r + 0.1, "Rozpora", ha="center", va="bottom", fontsize=10, color="grey")

    # Rysowanie nachylenia terenu
    if metoda_parcia.lower() == "coulomb" and theta != 0:
        theta_rad = math.radians(theta)
        x_terrain = [-1.5, 1.5]
        y_terrain = [0, (1.5 + 1.5) * math.tan(theta_rad)]
        ax.plot(x_terrain, y_terrain, color="green", linewidth=2, label="Teren nachylony")
    else:
        # Teren poziomy
        ax.plot([-1.5, 1.5], [0, 0], color="green", linewidth=2, label="Teren poziomy")

    # Rysowanie obciążenia zewnętrznego
    if q > 0:
        x_load = d  # Odległość od krawędzi wykopu
        ax.annotate(
            "",
            xy=(x_load, -0.5),
            xytext=(x_load, -1.5),
            arrowprops=dict(facecolor="red", shrink=0.05),
        )
        ax.text(x_load + 0.05, -1, f"q = {q} kPa", color="red", fontsize=10)

    # Ustawienia wykresu
    ax.set_xlim(-1.5, 2)
    ax.set_ylim(z + 1, -2)
    ax.axis("off")
    ax.legend()

    # Dostosowanie marginesów
    plt.tight_layout()

    # Konwersja wykresu do obrazu
    pngImage = io.BytesIO()
    fig.savefig(pngImage, format="png", bbox_inches="tight")
    pngImage.seek(0)
    plot_url = base64.b64encode(pngImage.getvalue()).decode("ascii")

    plt.close(fig)  # Zamknięcie wykresu, aby zwolnić pamięć

    return plot_url

@app.route('/docs')
def docs():
    return render_template('docs.html')

    
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    plot_url = None
    cross_section_plot_url = None
    form_data = None
    layer_pressures = []
    soil_parameters = []

    graphJSON = None
    plot_base64 = None

    NORMY_UZYTE = "PN-EN 1997-1:2008 Eurokod 7 – Projektowanie geotechniczne. Część 1: Zasady ogólne."

    if request.method == "POST":
        form_data = request.form.to_dict(flat=False)
        try:
            missing_fields = []

            # Pobranie i walidacja danych z formularza
            metoda_parcia = request.form.get("metoda_parcia")
            if not metoda_parcia:
                missing_fields.append("'Metoda obliczeń parcia gruntu'")

            # Pobranie dodatkowych parametrów dla metody Coulomba
            delta_input = request.form.get("delta", "0")
            beta_input = request.form.get("beta", "0")
            theta_input = request.form.get("theta", "0")

            # Pobranie list z formularza
            H_i = request.form.getlist("H_i[]")
            gamma_i = request.form.getlist("gamma_i[]")
            gamma_sat_i = request.form.getlist("gamma_sat_i[]")
            phi_i = request.form.getlist("phi_i[]")
            c_i = request.form.getlist("c_i[]")

            if not H_i or not gamma_i or not phi_i or not c_i:
                error = "Brak danych dla warstw gruntu. Upewnij się, że wprowadziłeś wszystkie wymagane wartości dla każdej warstwy."
                return render_template("index.html", error=error, form_data=form_data)

            # Pobranie pozostałych parametrów
            B_input = request.form.get("B")
            h_r_input = request.form.get("h_r")
            L_b_input = request.form.get("L_b")
            f_y_input = request.form.get("f_y")
            gamma_M_input = request.form.get("gamma_M")
            q_input = request.form.get("q")
            d_input = request.form.get("d")
            zw_input = request.form.get("zw")
            obudowa = request.form.get("obudowa")
            kotwienie = request.form.get("kotwienie")
            liczba_kotwien_input = request.form.get("liczba_kotwien")

            # Walidacja obecności wymaganych pól
            required_fields = {
                "Szerokość wykopu B": B_input,
                "Wysokość montażu rozpory h_r": h_r_input,
                "Długość blatu szalunkowego L_b": L_b_input,
                "Granica plastyczności stali f_y": f_y_input,
                "Współczynnik bezpieczeństwa materiału gamma_M": gamma_M_input,
                "Wielkość obciążenia q": q_input,
                "Odległość od krawędzi wykopu d": d_input,
                "Poziom wody gruntowej zw": zw_input,
                "Typ obudowy": obudowa,
                "Kotwienie": kotwienie,
            }

            for field_name, field_value in required_fields.items():
                if not field_value:
                    missing_fields.append(f"'{field_name}'")

            if kotwienie == "tak" and not liczba_kotwien_input:
                missing_fields.append("'Liczba kotwień'")

            if missing_fields:
                error = "Brak następujących pól w formularzu: " + ", ".join(missing_fields)
                return render_template("index.html", error=error, form_data=form_data)

            # Konwersja wartości na float
            try:
                B = float(B_input)
                h_r = float(h_r_input)
                L_b = float(L_b_input)
                f_y = float(f_y_input)
                gamma_M = float(gamma_M_input)
                q = float(q_input)
                d = float(d_input)
                zw = float(zw_input)
                gamma_w = 9.81
                delta = float(delta_input) if delta_input else 0.0
                beta = float(beta_input) if beta_input else 0.0
                theta = float(theta_input) if theta_input else 0.0
                liczba_kotwien = int(liczba_kotwien_input) if liczba_kotwien_input else 0
            except ValueError as e:
                error = "Wprowadzono niepoprawne dane numeryczne. Upewnij się, że wszystkie pola są wypełnione poprawnie liczbami."
                logging.error(f"ValueError during float conversion: {e}")
                return render_template("index.html", error=error, form_data=form_data)

            # Logowanie oryginalnych list
            logging.debug(f"Original Lists -> H_i: {H_i}, gamma_i: {gamma_i}, gamma_sat_i: {gamma_sat_i}, phi_i: {phi_i}, c_i: {c_i}")

            # Konwersja list wartości na listy liczb
            try:
                H_i = [float(h) for h in H_i]
                gamma_i = [float(g) for g in gamma_i]
                phi_i = [float(p) for p in phi_i]
                c_i = [float(c) for c in c_i]
                # Przetwarzanie gamma_sat_i z użyciem zip_longest
                from itertools import zip_longest

                gamma_sat_i_processed = []
                for i, (gs, g_i) in enumerate(zip_longest(gamma_sat_i, gamma_i)):
                    if gs:
                        try:
                            gs = float(gs)
                        except ValueError:
                            error = f"Niepoprawna wartość gamma_sat_i w warstwie {i+1}."
                            logging.error(error)
                            return render_template("index.html", error=error, form_data=form_data)
                        gamma_sat_i_processed.append(gs)
                    elif g_i is not None:
                        gamma_sat_i_processed.append(g_i)
                    else:
                        error = f"Brak wartości gamma_sat_i i gamma_i w warstwie {i+1}."
                        logging.error(error)
                        return render_template("index.html", error=error, form_data=form_data)

                # Uzupełnienie brakujących wartości, jeśli gamma_sat_i_processed jest krótsze niż H_i
                if len(gamma_sat_i_processed) < len(H_i):
                    missing = len(H_i) - len(gamma_sat_i_processed)
                    gamma_sat_i_processed += [gamma_i[-1]] * missing
                    logging.debug(f"Uzupełniono gamma_sat_i o {missing} wartości z gamma_i[-1]")

                gamma_sat_i = gamma_sat_i_processed

                logging.debug(f"Processed Lists -> H_i: {H_i}, gamma_i: {gamma_i}, gamma_sat_i: {gamma_sat_i}, phi_i: {phi_i}, c_i: {c_i}")

            except Exception as e:
                error = "Wystąpił błąd podczas przetwarzania danych warstw gruntu."
                logging.error(f"Exception during list processing: {e}")
                return render_template("index.html", error=error, form_data=form_data)

            # Sprawdzenie długości list
            if not (len(H_i) == len(gamma_i) == len(gamma_sat_i) == len(phi_i) == len(c_i)):
                error = "Nie wszystkie warstwy gruntu mają komplet danych. Upewnij się, że wprowadziłeś wszystkie wymagane wartości dla każdej warstwy."
                logging.error(f"List lengths do not match: H_i: {len(H_i)}, gamma_i: {len(gamma_i)}, gamma_sat_i: {len(gamma_sat_i)}, phi_i: {len(phi_i)}, c_i: {len(c_i)}")
                return render_template("index.html", error=error, form_data=form_data)

            # Całkowita głębokość wykopu
            H = sum(H_i)

            # Obliczenia dla każdej warstwy
            sigma_h_total = calculate_total_soil_pressure_with_surcharge_and_water(
                H_i,
                gamma_i,
                gamma_sat_i,
                phi_i,
                c_i,
                q,
                d,
                zw,
                gamma_w,
                metoda_parcia,
                delta,
                beta,
                theta,
            )

            # Update M_max calculation
            M_max = calculate_max_moment(sigma_h_total, H, liczba_kotwien)

            # Obliczenie sigma_v0 i u na głębokości H (dna wykopu)
            sigma_v0 = 0  # całkowite naprężenie pionowe na głębokości dna wykopu
            u = 0  # ciśnienie porowe na tej głębokości
            z = 0

            for i in range(len(H_i)):
                H_layer = H_i[i]
                gamma = gamma_i[i]
                gamma_sat = gamma_sat_i[i]

                if z >= H:
                    # Jesteśmy poniżej dna wykopu, przerywamy pętlę
                    break

                if z + H_layer > H:
                    # Warstwa kończy się poniżej dna wykopu
                    H_layer = H - z  # Bierzemy tylko część warstwy do głębokości H

                if z + H_layer <= zw:
                    # Warstwa w całości powyżej poziomu wody gruntowej
                    sigma_v0 += gamma * H_layer
                elif z >= zw:
                    # Warstwa w całości poniżej poziomu wody gruntowej
                    sigma_v0 += gamma_sat * H_layer
                    u += gamma_w * H_layer
                else:
                    # Warstwa częściowo powyżej i częściowo poniżej poziomu wody gruntowej
                    H_above = zw - z
                    H_below = H_layer - H_above
                    sigma_v0 += gamma * H_above + gamma_sat * H_below
                    u += gamma_w * H_below

                z += H_layer

            # Teraz sigma_v0 i u są obliczone do głębokości H
            sigma_p0 = sigma_v0 - u  # efektywne naprężenie pionowe na głębokości dna wykopu

            # Ciśnienie porowe po odwodnieniu (zakładamy, że woda została obniżona do dna wykopu)
            u_po = 0
            sigma_p0_po = sigma_v0 - u_po

            # Zmiana efektywnego naprężenia
            delta_sigma = sigma_p0_po - sigma_p0

            # Debugowanie
            logging.debug(f"sigma_v0: {sigma_v0:.2f}, u: {u:.2f}, sigma_p0: {sigma_p0:.2f}, u_po: {u_po:.2f}, sigma_p0_po: {sigma_p0_po:.2f}, delta_sigma: {delta_sigma:.2f}")

            settlement = None  # Inicjalizacja zmiennej
            settlement_mm = None

            # Pobranie parametrów do obliczeń osiadania
            e0_input = request.form.get("e0")
            Cc_input = request.form.get("Cc")
            H0_input = request.form.get("H0")

            if e0_input and Cc_input and H0_input:
                try:
                    e0 = float(e0_input)
                    Cc = float(Cc_input)
                    H0 = float(H0_input)
                    # Obliczenie osiadania
                    settlement = calculate_settlement(e0, delta_sigma, H0, Cc, sigma_p0)
                    settlement_mm = settlement * 1000  # Konwersja na milimetry
                    logging.debug(f"Settlement: {settlement_mm} mm")
                except ValueError:
                    settlement_mm = None
                    logging.error("Niepoprawne dane do obliczeń osiadania.")
            else:
                settlement_mm = None
                logging.debug("Brak danych do obliczeń osiadania.")

            # Obliczenie wskaźnika wytrzymałości przekroju W
            # Przyjmujemy wymiary przekroju dla szalunku słupowego ciężkiego
            if obudowa == "szalunek_slupowy_ciezki":
                t = 0.12  # Grubość blatu [m]
                b = 1.0  # Szerokość blatu [m]
                f_y = float(f_y_input)  # Granica plastyczności stali [MPa]
            elif obudowa == "szalunek_slupowy_lekki":
                t = 0.05
                b = 1.0
                f_y = float(f_y_input)
            elif obudowa == "szalunek_scianowy":
                t = 0.15
                b = 1.0
                f_y = 24  # Berlinka wytrzymałość 24 MPa
            else:
                # Domyślne wartości
                t = 0.12
                b = 1.0
                f_y = float(f_y_input)

            W = calculate_section_modulus(b, t)
            logging.debug(f"Section Modulus W: {W}")

            # Sprawdzenie warunku wytrzymałości na zginanie
            is_safe = check_bending_strength(M_max, W, f_y, gamma_M)
            logging.debug(f"Bending Strength Check: {'Safe' if is_safe else 'Unsafe'}")

            # Przygotowanie danych do wykresu
            z_values = [0]  # Start z = 0
            sigma_h_values = []
            layer_pressures = []
            soil_parameters = []
            z_accum = 0
            for i in range(len(H_i)):
                # Inicjalizacja parametrów warstwy
                phi = phi_i[i]
                c = c_i[i]
                gamma = gamma_i[i]
                gamma_sat = gamma_sat_i[i]
                H_layer = H_i[i]
                z_start = z_accum
                z_end = z_accum + H_layer
                z_values.append(z_end)

                # Wybór metody obliczeń
                if metoda_parcia.lower() == "coulomb":
                    Ka = calculate_Ka_coulomb(phi, delta, beta, theta)
                elif metoda_parcia.lower() == "terzaghi":
                    Ka = calculate_Ka_terzaghi(phi)
                elif metoda_parcia.lower() == "tschebotarioff":
                    Ka = calculate_Ka_tschebotarioff(phi)
                else:
                    Ka = calculate_Ka(phi)  # Domyślnie Rankine
                logging.debug(f"Metoda: {metoda_parcia}, Warstwa {i+1}, Ka: {Ka:.4f}")

                # Obliczanie parcia gruntu
                try:
                    sigma_start = calculate_soil_pressure_with_water(
                        Ka, gamma, gamma_sat, gamma_w, z_start, zw, c
                    )
                    sigma_end = calculate_soil_pressure_with_water(
                        Ka, gamma, gamma_sat, gamma_w, z_end, zw, c
                    )
                except Exception as e:
                    logging.error(f"Błąd podczas obliczania ciśnienia gruntu dla warstwy {i+1}: {e}")
                    error = f"Wystąpił błąd podczas obliczania ciśnienia gruntu dla warstwy {i+1}."
                    return render_template("index.html", error=error, form_data=form_data)

                # Dodawanie wartości sigma_h dla parcia czynnego
                sigma_h_values.append(sigma_start)
                sigma_h_values.append(sigma_end)

                # Dodanie parcia od obciążenia zewnętrznego
                reduction_start = reduction_factor(d, z_start)
                reduction_end = reduction_factor(d, z_end)
                sigma_start += calculate_surcharge_pressure(Ka, q, reduction_start)
                sigma_end += calculate_surcharge_pressure(Ka, q, reduction_end)

                # Średnie parcie w warstwie
                sigma_avg = (sigma_start + sigma_end) / 2

                # Dodawanie dodatkowych danych do wykresów
                

                # Zbieranie danych do raportu
                layer_pressures.append(
                    {
                        "layer": i + 1,
                        "depth_start": round(z_start, 2),
                        "depth_end": round(z_end, 2),
                        "sigma_start": round(sigma_start, 2),
                        "sigma_end": round(sigma_end, 2),
                        "sigma_avg": round(sigma_avg, 2),
                    }
                )
                # Soil parameters for table
                soil_parameters.append(
                    {
                        "layer": i + 1,
                        "H_i": H_i[i],
                        "gamma_i": gamma_i[i],
                        "gamma_sat_i": gamma_sat_i[i],
                        "phi_i": phi_i[i],
                        "c_i": c_i[i],
                    }
                )

                logging.debug(f"Layer {i+1} -> H_i: {H_i[i]}, gamma_i: {gamma_i[i]}, gamma_sat_i: {gamma_sat_i[i]}, phi_i: {phi_i[i]}, c_i: {c_i[i]}")
                logging.debug(f"Layer {i+1} -> sigma_start: {sigma_start}, sigma_end: {sigma_end}, sigma_avg: {sigma_avg}")

                z_accum = z_end


                

            logging.debug(f"z_values: {z_values}")
            logging.debug(f"sigma_h_values: {sigma_h_values}")

            # Przygotowanie wyniku
            result = {
                "norms_used": "PN-EN 1997-1:2008 Eurokod 7 – Projektowanie geotechniczne",
                "calculation_method": metoda_parcia.capitalize(),
                "safety_factor": round(gamma_M, 2),
                "shoring_parameters": {
                    "obudowa": obudowa.replace("_", " ").capitalize(),
                    "h_r": round(h_r, 2),
                    "L_b": round(L_b, 2),
                    "f_y": round(f_y, 2),
                },
                "sigma_h_total": round(sigma_h_total, 2),
                "M_max": round(M_max, 2),
                "W": round(W, 6),
                "is_safe": is_safe,
                "settlement": round(settlement_mm, 2) if settlement_mm else None,
            }

            # Generowanie interaktywnego wykresu
            graphJSON, plot_base64 = create_pressure_plot(
                z_values,
                sigma_h_values,
                result.get("calculation_method"),
                phi_i,
                gamma_i,
                gamma_sat_i,
                c_i,
                H_i,
                zw,
            )
            plot_url = plot_base64

            # Upewnij się, że 'theta' jest zdefiniowana
            logging.debug(f"Theta przed wywołaniem create_cross_section_plot: {theta}")
            logging.debug(f"z_values: {z_values}")
            logging.debug(f"sigma_h_values: {sigma_h_values}")


            cross_section_plot_url = create_cross_section_plot(
                H_i, zw, H, q, d, theta, h_r, metoda_parcia, z_values, sigma_h_values
            )

            # Debugowanie
            if plot_base64:
                logging.debug(f"plot_base64 length: {len(plot_base64)}")
            if cross_section_plot_url:
                logging.debug(f"cross_section_plot_url length: {len(cross_section_plot_url)}")

            # Przechowywanie danych w hidden fields do pobrania raportu
            return render_template(
                "index.html",
                result=result,
                graphJSON=graphJSON,
                plot_url=plot_base64,
                plot_base64=plot_base64,  # Dodatkowe przekazanie
                cross_section_plot_url=cross_section_plot_url,
                error=error,
                form_data=form_data,
                layer_pressures=layer_pressures,
                soil_parameters=soil_parameters,
            )
        except Exception as e:
            logging.error(f"Błąd podczas przetwarzania formularza: {e}", exc_info=True)
            error = "Wystąpił błąd podczas przetwarzania formularza. Upewnij się, że wszystkie pola są wypełnione poprawnie."
            return render_template("index.html", error=error, form_data=form_data)
    else:
        form_data = None
        result = None
        graphJSON = None
        plot_base64 = None
        cross_section_plot_url = None
        layer_pressures = []
        soil_parameters = []
        error = None

    return render_template(
        "index.html",
        result=result,
        graphJSON=graphJSON,
        plot_url=plot_base64,
        plot_base64=plot_base64,  # Dodatkowe przekazanie
        cross_section_plot_url=cross_section_plot_url,
        error=error,
        form_data=form_data,
        layer_pressures=layer_pressures,
        soil_parameters=soil_parameters,
    )

@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        # Pobranie danych z formularza
        result_json = request.form.get('result')
        plot_url = request.form.get('plot_url')  # Upewnij się, że nazwa jest 'plot_url'
        soil_parameters_json = request.form.get('soil_parameters')
        cross_section_plot_url = request.form.get('cross_section_plot_url')
        watermark = request.form.get('watermark')
        password = request.form.get('password')  # Używane tylko gdy watermark == 'no'

        # Logowanie wartości
        logging.debug(f"Received plot_url: {plot_url[:50] if plot_url else 'None'}")

        # Logowanie wartości
        if plot_url:
            logging.debug(f"Received plot_url: {plot_url[:50]}...")
        else:
            logging.debug("Received plot_url is None")
        if cross_section_plot_url:
            logging.debug(f"Received cross_section_plot_url: {cross_section_plot_url[:50]}...")
        else:
            logging.debug("Received cross_section_plot_url is None")
        if result_json:
            logging.debug(f"Received result_json: {result_json[:50]}...")
        else:
            logging.debug("Received result_json is None")
        if soil_parameters_json:
            logging.debug(f"Received soil_parameters_json: {soil_parameters_json[:50]}...")
        else:
            logging.debug("Received soil_parameters_json is None")

        # Walidacja obecności danych
        if not result_json or not soil_parameters_json:
            flash("Brak danych do wygenerowania raportu.", "danger")
            return redirect(url_for('index'))

        # Dekodowanie danych JSON
        try:
            result = json.loads(result_json)
            soil_parameters = json.loads(soil_parameters_json)
        except json.JSONDecodeError:
            flash("Niepoprawny format danych do raportu.", "danger")
            return redirect(url_for('index'))

        # Weryfikacja hasła, jeśli watermark jest 'no'
        if watermark == 'no':
            if password not in CORRECT_PASSWORDS:
                flash("Niepoprawne hasło. Raport będzie zawierał znak wodny.", "warning")
                watermark = 'yes'  # Zmieniamy na 'yes', aby dodać znak wodny

        # Renderowanie szablonu HTML raportu
        rendered = render_template(
            'report.html',
            result=result,
            plot_url=plot_url,  # Przekazujemy plot_url
            soil_parameters=soil_parameters,
            cross_section_plot_url=cross_section_plot_url,
            watermark=watermark
        )

        # Opcje CSS
        css_files = ['css/report.css']
        if watermark == 'yes':
            css_files.append('css/watermark.css')

         # Opcje PDF
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None
        }    

        # Pobranie pełnych ścieżek do plików CSS
        stylesheets = []
        for css_file in css_files:
            css_path = os.path.join(os.path.dirname(__file__), 'static', css_file)
            if os.path.exists(css_path):
                stylesheets.append(CSS(filename=css_path))
            else:
                app.logger.warning(f"Plik CSS nie istnieje: {css_path}")

        # Generowanie PDF za pomocą WeasyPrint
        html = HTML(string=rendered, base_url=os.path.join(os.path.dirname(__file__), 'static'))
        pdf = html.write_pdf(stylesheets=stylesheets)

        # Zwrot PDF jako plik do pobrania
        return send_file(
            io.BytesIO(pdf),
            download_name='raport.pdf',
            as_attachment=True,
            mimetype='application/pdf'
        )

    except Exception as e:
        app.logger.error(f"Error generating PDF: {e}", exc_info=True)
        flash(f"Nie udało się wygenerować raportu: {e}", "danger")
        return redirect(url_for('index'))






if __name__ == "__main__":
    app.run(debug=True)
