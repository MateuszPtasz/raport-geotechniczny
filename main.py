from dotenv import load_dotenv

load_dotenv()


import requests
import stat
import tarfile

import os
import math
import io
import json
import base64
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from flask import send_file
from calculations import (
    calculate_Ka,
    calculate_Ka_coulomb,
    calculate_Ka_terzaghi,
    calculate_Ka_tschebotarioff,
    calculate_settlement,
    calculate_soil_pressure_with_water,
    calculate_total_soil_pressure_with_surcharge_and_water,
    calculate_max_moment,
    check_bending_strength,
    calculate_section_modulus,
    calculate_surcharge_pressure,
    reduction_factor,
)


from PIL import Image
from weasyprint import HTML, CSS

import plotly.graph_objs as go
import plotly

# Zakładając, że folder 'fonts' znajduje się w tym samym katalogu co 'main.py'
FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")

app = Flask(__name__)
app.config["SECRET_KEY"] = "PRB123"  # Zmień na silny, losowy klucz

# Konfiguracja logowania
logging.basicConfig(level=logging.DEBUG)


# Ładowanie listy haseł z zmiennej środowiskowej
CORRECT_PASSWORDS = os.getenv("REPORT_PASSWORDS", "default_password").split(",")
# Usunięcie nadmiarowych spacji w hasłach
CORRECT_PASSWORDS = [pwd.strip() for pwd in CORRECT_PASSWORDS]

logging.debug(f"CORRECT_PASSWORDS loaded: {CORRECT_PASSWORDS}")

# Upewnij się, że pliki czcionek są w katalogu aplikacji
pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", "DejaVuSans-Bold.ttf"))


def get_phi_at_depth(z, H_i, phi_values):
    """
    Zwraca wartość kąta tarcia wewnętrznego φ dla danej głębokości z.
    """
    z_accum = 0
    for i, H_layer in enumerate(H_i):
        z_accum += H_layer
        if z <= z_accum:
            return phi_values[i]
    return phi_values[
        -1
    ]  # Jeśli z jest większe niż całkowita głębokość, zwróć ostatnią wartość


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
    return gamma_values[
        -1
    ]  # Jeśli z jest większe niż całkowita głębokość, zwróć ostatnią wartość


def get_c_at_depth(z, H_i, c_values):
    """
    Zwraca wartość spójności c dla danej głębokości z.
    """
    z_accum = 0
    for i, H_layer in enumerate(H_i):
        z_accum += H_layer
        if z <= z_accum:
            return c_values[i]
    return c_values[
        -1
    ]  # Jeśli z jest większe niż całkowita głębokość, zwróć ostatnią wartość


def create_pressure_plot(
    z_values,
    sigma_h_values,
    calculation_method,
    phi_values,
    gamma_values,
    gamma_sat_values,
    c_values,
    H_i,
    zw,
):
    """
    Tworzy wykres rozkładu parcia gruntu za pomocą Plotly i zwraca go w formie JSON oraz base64.
    """
    print("Tworzenie wykresu interaktywnego...")

    # Debugowanie: sprawdzenie długości list
    print(
        f"len(z_values): {len(z_values)}, len(gamma_values): {len(gamma_values)}, len(gamma_sat_values): {len(gamma_sat_values)}"
    )

    # Obliczenie parcia biernego
    sigma_h_passive = []
    for i in range(len(z_values)):
        z = z_values[i]
        phi = get_phi_at_depth(z, H_i, phi_values)
        gamma = get_gamma_at_depth(z, H_i, gamma_values, gamma_sat_values, zw)
        c = get_c_at_depth(z, H_i, c_values)
        Kp = (1 + math.sin(math.radians(phi))) / (1 - math.sin(math.radians(phi)))

        # Obliczanie gamma_sub na podstawie głębokości
        if z > zw:
            gamma_sub = gamma - 9.81  # γ' = γ_sat - γ_w
        else:
            gamma_sub = gamma  # Powyżej poziomu wody gruntowej

        sigma_v = gamma_sub * z  # Naprężenie pionowe efektywne
        sigma_h_p = Kp * sigma_v + 2 * c * math.sqrt(Kp)
        sigma_h_passive.append(sigma_h_p)

    # Ustawienie większych rozmiarów figury Plotly
    fig = go.Figure(
        layout=dict(width=800, height=600)
    )  # Zwiększono szerokość i wysokość

    # Dodajemy parcie czynne
    fig.add_trace(
        go.Scatter(
            x=sigma_h_values,
            y=z_values,
            mode="lines+markers",
            name="Parcie czynne",
            hovertemplate="Parcie czynne: %{x:.2f} kPa<br>"
            + "Głębokość: %{y:.2f} m<br>"
            + "<extra></extra>",
        )
    )

    # Dodajemy parcie bierne
    fig.add_trace(
        go.Scatter(
            x=sigma_h_passive,
            y=z_values,
            mode="lines+markers",
            name="Parcie bierne",
            hovertemplate="Parcie bierne: %{x:.2f} kPa<br>"
            + "Głębokość: %{y:.2f} m<br>"
            + "<extra></extra>",
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
        # Kodowanie obrazu PNG w base64
        plot_base64 = base64.b64encode(plot_png).decode("utf-8")
        print("Wykres PNG został pomyślnie wygenerowany.")
        print(f"plot_base64 length: {len(plot_base64)}")
    except Exception as e:
        print(f"Błąd podczas generowania obrazu wykresu: {e}")
        plot_base64 = None

    return graphJSON, plot_base64


def create_cross_section_plot(
    H_i, zw, H_total, q, d, theta, h_r, metoda_parcia, z_values, sigma_h_values
):
    """
    Tworzy przekrój poprzeczny wykopu za pomocą Matplotlib i zwraca go w formie base64.
    """
    from matplotlib.patches import Polygon

    # Utworzenie figury
    fig, ax = plt.subplots(figsize=(6, 8))

    # Rysowanie warstw gruntu po lewej stronie wykopu
    z = 0
    for i, H_layer in enumerate(H_i):
        rect = plt.Rectangle(
            (-1, z), 1, H_layer, facecolor=f"C{i%10}", edgecolor="black"
        )
        ax.add_patch(rect)
        ax.text(
            -0.5,
            z + H_layer / 2,
            f"Warstwa {i+1}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=10,
        )
        z += H_layer

    # Rysowanie poziomu wody gruntowej
    ax.axhline(y=zw, color="blue", linestyle="--", label="Poziom wody gruntowej")
    ax.text(
        -1.2,
        zw,
        "Poziom wody gruntowej",
        ha="right",
        va="center",
        fontsize=10,
        color="blue",
    )

    # Rysowanie szalunku na x = 0
    ax.plot([0, 0], [0, z], color="brown", linewidth=5, label="Szalunek")
    ax.text(
        0.05,
        z / 2,
        "Szalunek",
        ha="left",
        va="center",
        rotation=90,
        fontsize=10,
        color="brown",
    )

    # Rysowanie rozpory
    rozpora_coords = [
        (0, h_r - 0.05),
        (0, h_r + 0.05),
        (1.0, h_r + 0.05),
        (1.0, h_r - 0.05),
    ]

    rozpora = Polygon(rozpora_coords, closed=True, facecolor="grey", edgecolor="black")
    ax.add_patch(rozpora)
    ax.text(
        0.5, h_r + 0.1, "Rozpora", ha="center", va="bottom", fontsize=10, color="grey"
    )

    # Rysowanie nachylenia terenu
    if metoda_parcia.lower() == "coulomb" and theta != 0:
        theta_rad = math.radians(theta)
        x_terrain = [-1.5, 1.5]
        y_terrain = [0, (1.5 + 1.5) * math.tan(theta_rad)]
        ax.plot(
            x_terrain, y_terrain, color="green", linewidth=2, label="Teren nachylony"
        )
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


@app.route("/", methods=["GET", "POST"])
def process_form():
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
                error = "Brak następujących pól w formularzu: " + ", ".join(
                    missing_fields
                )
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
                liczba_kotwien = (
                    int(liczba_kotwien_input) if liczba_kotwien_input else 0
                )
            except ValueError as e:
                error = "Wprowadzono niepoprawne dane numeryczne. Upewnij się, że wszystkie pola są wypełnione poprawnie liczbami."
                return render_template("index.html", error=error, form_data=form_data)

            # Konwersja list wartości na listy liczb
            try:
                H_i = [float(h) for h in H_i]
                gamma_i = [float(g) for g in gamma_i]
                gamma_sat_i = [
                    float(gs) if gs else gamma_i[i] for i, gs in enumerate(gamma_sat_i)
                ]
                # Uzupełnij brakujące wartości, jeśli gamma_sat_i jest krótsze niż H_i
                if len(gamma_sat_i) < len(H_i):
                    gamma_sat_i += [gamma_i[-1]] * (len(H_i) - len(gamma_sat_i))
                phi_i = [float(p) for p in phi_i]
                c_i = [float(c) for c in c_i]
            except ValueError as e:
                error = "Wprowadzono niepoprawne dane w warstwach gruntu. Upewnij się, że wszystkie pola są wypełnione poprawnie liczbami."
                return render_template("index.html", error=error, form_data=form_data)

            # Walidacja zakresów wartości
            for value in H_i:
                if value <= 0:
                    error = "Grubość warstwy H_i musi być większa niż 0."
                    return render_template(
                        "index.html", error=error, form_data=form_data
                    )
            for value in gamma_i:
                if value <= 0:
                    error = "Ciężar objętościowy γ_i musi być większy niż 0."
                    return render_template(
                        "index.html", error=error, form_data=form_data
                    )
            for value in phi_i:
                if value < 0 or value > 90:
                    error = "Kąt tarcia wewnętrznego φ_i musi być między 0 a 90 stopni."
                    return render_template(
                        "index.html", error=error, form_data=form_data
                    )
            for value in c_i:
                if value < 0:
                    error = "Spójność c_i nie może być ujemna."
                    return render_template(
                        "index.html", error=error, form_data=form_data
                    )

            # Sprawdzenie długości list
            if not (
                len(H_i) == len(gamma_i) == len(gamma_sat_i) == len(phi_i) == len(c_i)
            ):
                error = "Nie wszystkie warstwy gruntu mają komplet danych. Upewnij się, że wprowadziłeś wszystkie wymagane wartości dla każdej warstwy."
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
            sigma_p0 = (
                sigma_v0 - u
            )  # efektywne naprężenie pionowe na głębokości dna wykopu

            # Ciśnienie porowe po odwodnieniu (zakładamy, że woda została obniżona do dna wykopu)
            u_po = 0
            sigma_p0_po = sigma_v0 - u_po

            # Zmiana efektywnego naprężenia
            delta_sigma = sigma_p0_po - sigma_p0

            # Debugowanie
            print(
                f"sigma_v0: {sigma_v0:.2f}, u: {u:.2f}, sigma_p0: {sigma_p0:.2f}, u_po: {u_po:.2f}, sigma_p0_po: {sigma_p0_po:.2f}, delta_sigma: {delta_sigma:.2f}"
            )

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
                except ValueError:
                    settlement_mm = None
            else:
                settlement_mm = None

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

            # Sprawdzenie warunku wytrzymałości na zginanie
            is_safe = check_bending_strength(M_max, W, f_y, gamma_M)

            # Przygotowanie danych do wykresu
            z_values = []
            sigma_h_values = []
            layer_pressures = []
            z_accum = 0
            for i in range(len(H_i)):
                phi = phi_i[i]
                c = c_i[i]
                gamma = gamma_i[i]
                gamma_sat = gamma_sat_i[i]
                H_layer = H_i[i]
                z_start = z_accum
                z_end = z_accum + H_layer
                z_mid = (z_start + z_end) / 2

                # Wybór metody obliczeń
                if metoda_parcia.lower() == "coulomb":
                    Ka = calculate_Ka_coulomb(phi, delta, beta, theta)
                elif metoda_parcia.lower() == "terzaghi":
                    Ka = calculate_Ka_terzaghi(phi)
                elif metoda_parcia.lower() == "tschebotarioff":
                    Ka = calculate_Ka_tschebotarioff(phi)
                else:
                    Ka = calculate_Ka(phi)  # Domyślnie Rankine
                print(f"Metoda: {metoda_parcia}, Warstwa {i+1}, Ka: {Ka:.4f}")

                # Obliczanie parcia gruntu
                sigma_start = calculate_soil_pressure_with_water(
                    Ka, gamma, gamma_sat, gamma_w, z_start, zw, c
                )
                sigma_end = calculate_soil_pressure_with_water(
                    Ka, gamma, gamma_sat, gamma_w, z_end, zw, c
                )

                # Dodanie parcia od obciążenia zewnętrznego
                reduction_start = reduction_factor(d, z_start)
                reduction_end = reduction_factor(d, z_end)
                sigma_start += calculate_surcharge_pressure(Ka, q, reduction_start)
                sigma_end += calculate_surcharge_pressure(Ka, q, reduction_end)

                # Średnie parcie w warstwie
                sigma_avg = (sigma_start + sigma_end) / 2

                # Calculate soil pressure at start and end of the layer
                z_points = [z_accum, z_accum + H_i[i]]
                for z_point in z_points:
                    sigma_h = calculate_soil_pressure_with_water(
                        Ka, gamma, gamma_sat, gamma_w, z_point, zw, c
                    )
                    # Add surcharge pressure
                    reduction = reduction_factor(d, z_point)
                    sigma_h += calculate_surcharge_pressure(Ka, q, reduction)

                    z_values.append(z_point)
                    sigma_h_values.append(sigma_h)

                z_accum += H_i[i]

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

                z_accum = z_end

            print("z_values:", z_values)
            print("sigma_h_values:", sigma_h_values)

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
                result.get(
                    "calculation_method"
                ),  # używamy result['calculation_method']
                phi_i,
                gamma_i,
                gamma_sat_i,
                c_i,
                H_i,
                zw,
            )

            # Upewnij się, że 'theta' jest zdefiniowana
            print(f"Theta przed wywołaniem create_cross_section_plot: {theta}")

            cross_section_plot_url = create_cross_section_plot(
                H_i, zw, H, q, d, theta, h_r, metoda_parcia, z_values, sigma_h_values
            )

            # Debugowanie
            print("plot_base64:", plot_base64[:50] if plot_base64 else None)
            print(
                "cross_section_plot_url:",
                cross_section_plot_url[:50] if cross_section_plot_url else None,
            )

            # Przechowywanie danych w hidden fields do pobrania raportu
            return render_template(
                "index.html",
                result=result,
                graphJSON=graphJSON,
                plot_url=plot_base64,
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
        cross_section_plot_url=cross_section_plot_url,
        error=error,
        form_data=form_data,
        layer_pressures=layer_pressures,
        soil_parameters=soil_parameters,
    )


@app.route("/download_report", methods=["POST"])
def generate_report():
    try:
        # Pobranie danych z formularza
        result_json = request.form.get("result")
        plot_base64 = request.form.get("plot_base64")
        soil_parameters_json = request.form.get("soil_parameters")
        cross_section_plot_url = request.form.get("cross_section_plot_url")
        watermark = request.form.get("watermark")
        password = request.form.get("password")  # Używane tylko gdy watermark == 'no'

        # Dekodowanie danych JSON
        result = json.loads(result_json)
        soil_parameters = json.loads(soil_parameters_json)

        # Weryfikacja hasła, jeśli watermark jest 'no'
        if watermark == "no":
            if password not in CORRECT_PASSWORDS:
                flash(
                    "Niepoprawne hasło. Raport będzie zawierał znak wodny.", "warning"
                )
                watermark = "yes"  # Zmieniamy na 'yes', aby dodać znak wodny

        # Renderowanie szablonu HTML raportu
        rendered = render_template(
            "report.html",
            result=result,
            plot_base64=plot_base64,
            soil_parameters=soil_parameters,
            cross_section_plot_url=cross_section_plot_url,
            watermark=watermark,
        )

        # Opcje CSS
        css = CSS(
            string="""
            @page {
                size: A4;
                margin: 0.5in;
            }
            body {
                font-family: 'DejaVu Sans', sans-serif;
                margin: 0;
                padding: 0;
            }
            h1, h2 {
                text-align: center;
                word-wrap: break-word;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            table, th, td {
                border: 1px solid black;
            }
            th, td {
                padding: 8px;
                text-align: center;
                word-wrap: break-word;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            footer {
                margin-top: 50px;
                font-size: 10px;
                text-align: center;
                color: gray;
            }
            img {
                max-width: 100%;
                height: auto;
            }
        """
        )

        if watermark == "yes":
            # Dodanie znaku wodnego poprzez CSS
            css += CSS(
                string="""
                body::before {
                    content: "WODA";
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%) rotate(-45deg);
                    font-size: 100px;
                    color: rgba(200, 200, 200, 0.3);
                    z-index: -1;
                }
            """
            )

        # Generowanie PDF za pomocą WeasyPrint
        html = HTML(string=rendered, base_url=request.base_url)
        pdf = html.write_pdf(stylesheets=[css])

        # Zwrot PDF jako plik do pobrania
        return send_file(
            io.BytesIO(pdf),
            attachment_filename="raport.pdf",
            as_attachment=True,
            mimetype="application/pdf",
        )
    except Exception as e:
        app.logger.error(f"Error generating PDF: {e}", exc_info=True)
        flash(f"Nie udało się wygenerować raportu: {e}", "danger")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
