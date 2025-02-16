# Obliczanie Wytrzymałości Wykopów z Szalunkiem

## Opis

Aplikacja służy do wspomagania pracy przy projektowaniu komór zabezpieczonych szalunkiem. Umożliwia przeliczanie, czy dana komora (szalunek) typu ciężkiego wytrzyma obciążenia i będzie bezpieczna, oraz czy obciążenie poruszające się w pobliżu wykopu nie spowoduje zagrożenia dla ludzi wewnątrz. Aplikacja uwzględnia parametry geologiczne warstw gruntu, obciążenia zewnętrzne, parametry szalunku (blatu i rozpory), poziom wody gruntowej, a także umożliwia wybór metody obliczeniowej (Coulomb, Terzaghi-Peck, Tschebotarioff, Rankine). Na końcu obliczeń aplikacja określa, czy konstrukcja jest bezpieczna, rysuje wykres parcia gruntu i umożliwia zapisanie wyników w formacie PDF.

## Funkcje

*   Obliczanie parcia czynnego gruntu z uwzględnieniem wielu warstw gruntu, obciążenia naziomu, i poziomu wody gruntowej.
*   Wybór metody obliczeniowej:
    *   Coulomb (z uwzględnieniem kąta tarcia grunt-ściana i nachylenia terenu)
    *   Terzaghi-Peck (z korektami dla gruntów spoistych)
    *   Tschebotarioff (z uwzględnieniem współczynnika redukcyjnego)
    *   Rankine (uproszczona, dla poziomego terenu i braku tarcia grunt-ściana)
*   Obliczanie maksymalnego momentu zginającego.
*   Sprawdzanie nośności na zginanie dla szalunku.
*   Obliczanie osiadania gruntu (opcjonalnie, po wprowadzeniu dodatkowych parametrów).
*   Generowanie wykresu parcia gruntu.
*   Generowanie raportu PDF.

## Technologie

*   Python
*   Flask
*   ReportLab (do generowania PDF)
*   PyPDF2 (do łączenia PDF i dodawania znaku wodnego)
*   Matplotlib (do generowania wykresów)
*   Plotly (do generowania interaktywnych wykresów)
*   pdfkit (do konwersji HTML do PDF)
*   HTML, CSS, JavaScript

## Instalacja

1.  Sklonuj repozytorium:
    ```bash
    git clone https://github.com/MateuszPtasz/raport-geotechniczny
    ```
2.  Przejdź do katalogu projektu:
    ```bash
     cd raport-geotechniczny
    ```
3.  Zainstaluj wymagane biblioteki Pythona:
    ```bash
    pip install -r requirements.txt
    ```
   (Utwórz plik `requirements.txt` w głównym katalogu projektu i umieść w nim poniższe zależności):
    ```
    Flask==2.2.2
    python-dotenv
    reportlab
    PyPDF2
    matplotlib
    plotly
    pdfkit
    ```
4.  Zainstaluj wkhtmltopdf:
    *   **Windows:** Pobierz i zainstaluj `wkhtmltopdf` ze strony [https://wkhtmltopdf.org/downloads.html](https://wkhtmltopdf.org/downloads.html). Upewnij się, że ścieżka do `wkhtmltopdf.exe` jest poprawnie ustawiona w zmiennej środowiskowej `PATH` lub w konfiguracji aplikacji (tak jak masz to zrobione teraz).
5. Utwórz plik `makefile.env` (lub dostosuj, jeśli już go masz) i dodaj swoje zmienne środowiskowe, w tym `REPORT_PASSWORDS`.

## Uruchomienie

1.  Uruchom aplikację:
    ```bash
    python app.py
    ```
2.  Otwórz przeglądarkę i przejdź do adresu `http://127.0.0.1:5000/`.

## Autor

Mateusz Ptaszkowski ptaszkowski.mateusz@gmail.com

## Licencja

MIT License

Copyright (c) 2024 Mateusz Ptasznik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.