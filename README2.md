<img src="atmoscol.jpg" alt="thumbnail" width="800"/>

# Taller de datos científicos con Python y R - AtmosCol 2023

[![nightly-build](https://github.com/aladinor/Atmoscol2023/actions/workflows/nightly-build.yaml/badge.svg)](https://github.com/aladinor/Atmoscol2023/actions/workflows/nightly-build.yaml)
[![Binder](https://binder.projectpythia.org/badge_logo.svg)](https://binder.projectpythia.org/v2/gh/ProjectPythia/cookbook-template/main?labpath=notebooks)
[![DOI](https://zenodo.org/badge/686482876.svg)](https://zenodo.org/doi/10.5281/zenodo.8316796)

## Motivación

Este taller de iniciación a la programación científica con Python y R - AtmosCol 2023 tiene como objetivo promover el paradigma emergente
de investigación conocido como **'ciencia abierta'**. Este enfoque busca fomentar el acceso y la inclusión a los datos
hidrometeorológicos de diversas fuentes, así como la reproducibilidad de los códigos, con el fin de impulsar el
desarrollo colaborativo y la participación en actividades científicas en todos los niveles de la sociedad.

La ciencia abierta aboga por la **transparencia** y la **colaboración** en la investigación científica, fomentando la
**disponibilidad** de datos científicos, la capacidad de **reproducir los resultados**, y la **inclusión** de diversos sectores
de la **sociedad** en el proceso de investigación. Además, promueve la **comunicación efectiva** de los resultados científicos
y la **divulgación del conocimiento** en beneficio de la comunidad en general.

En el marco de este taller, se capacitará a los participantes en el uso de herramientas poderosas como `Python` y `R` para
trabajar con datos hidrometeorológicos y llevar a cabo **análisis científicos**. De esta manera, se empoderará a los
asistentes para contribuir de manera efectiva a la **investigación científica abierta**, lo que puede tener un impacto
significativo en el avance de la ciencia y en la **toma de decisiones informadas en Colombia**.

## Autores

[Alfonso Ladino-Rincon](https://github.com/aladinor)
[Nicole Rivera](https://github.com/nicolerivera1)
[Max Grover](https://github.com/mgrover1)

### Colaboradores

<a href="https://github.com/aladinor/Atmoscol2023/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=aladinor/Atmoscol2023" />
</a>

## Estructura

El taller estará compuesto por dos sesiones. En la sesión de la mañana trabajeremos con `Python` y acceso a los datos hidrometeorológicos de diversas fuentes. En la sesión de la tarde trabajaremos anális de series de tiempo usando `R`.

### Sección 1. Acceso a los datos hidrometeorológicos usando Python

|        Hora         |                                                          Contenido                                                           |                                   Tutor                                   |  Duración  |
| :-----------------: | :--------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------: | :--------: |
| 08:00 AM - 08:15 AM |            Apertura del curso. Arranque del Jupyter Lab, instalación de librerias y requerimientos para el taller            | Alfonso Ladino, Nicole Rivera, Nestor Bernal, Iván Arias, Maria F. Moreno | 15 minutos |
| 08:15 AM - 09:00 AM |                                    Introducción a Numpy, Pandas, Xarray, Py-Art y Xradar                                     |                              Alfonso Ladino                               | 45 minutos |
| 09:00 AM - 09:30 AM |            Acceso a los datos de estaciones IDEAM usando el portal de [datos abiertos](https://www.datos.gov.co/)            |                              Alfonso Ladino                               | 30 minutos |
| 09:30 AM - 10:00 AM | Acceso a los datos de [radares meteorológicos](https://registry.opendata.aws/ideam-radares/) de IDEAM usando Xradar y Py-Art |                              Alfonso Ladino                               | 30 minutos |
| 10:00 AM - 10:30 AM |                                                   Pausa para el refrigerio                                                   |                                                                           | 30 minutos |
| 10:30 AM - 11:00 AM |                                   Acceso a los datos de NASA (OPENDAP) y los datos de CMIP                                   |                       Alfonso Ladino, Nicole Rivera                       | 30 minutos |
| 11:00 AM - 11:30 PM |                                     Cálculo de la anomalia ENSO en el Pacífico Tropical                                      |                               Nicole Rivera                               | 30 minutos |
| 11:30 AM - 12:00 PM |                                      Gráficas del IPCC - Escenarios de Cambio Climático                                      |                               Nicole Rivera                               | 30 minutos |
| 12:00 PM - 01:30 PM |                                                           Almuerzo                                                           |                                                                           | 1.5 horas  |

### Sección 2. Anális de series de tiempo usando R

|        Hora         |                             Contenido                              |     Tutor     | Duración  |
| :-----------------: | :----------------------------------------------------------------: | :-----------: | :-------: |
| 01:30 PM - 03:00 PM | Homogenización de series de tiempo mensuales de precipitación en R | Néstor Bernal | 1.5 horas |

## Ejecutar los Notebooks

Pueden ejecutar los `notebooks` bien sea usando [Binder](https://mybinder.org/) o localmente en sus maquinas.

### Binder

La forma más sencilla de interactuar con un `Jupyter Notebook` es a través de [Binder](https://binder.projectpythia.org/), que permite la ejecución de un [Jupyter Book](https://jupyterbook.org) en la nube. Los detalles de cómo funciona `binder` no son muy relevantes por ahora. Todo lo que necesitamos saber es cómo iniciar un capítulo de Pythia Cookbooks a través de Binder. Simplemente navegue con el mouse hasta la esquina superior derecha del capítulo del libro que está viendo y haga clic en el ícono del cohete y asegúrese de seleccionar "iniciar Binder". Después de un momento, se te presentará un `Jupyter Lab` con el que podrás interactuar. Es decir. Podrás ejecutar e incluso cambiar los programas de ejemplo. Verás que las celdas de código no tienen salida al principio, hasta que las ejecutes presionando <kbd>Shift</kbd>+<kbd>Enter</kbd>. Los detalles completos sobre cómo interactuar con un cuaderno Jupyter activo se describen en [Introducción a Jupyter](https://foundations.projectpythia.org/foundations/getting-started-jupyter.html).

### Ejecutar de manera local

Si está interesado en ejecutar este material localmente en su computadora, deberá seguir este flujo de trabajo:

1. Clone el repositorio `https://github.com/aladinor/Atmoscol2023.git` usando el siguiente comando de consola:

   ```bash
    git clone https://github.com/aladinor/Atmoscol2023.git
   ```

1. Entre en la carpeta de `Atmoscol2023`
   ```bash
   cd Atmoscol2023
   ```
1. Cree y active su ambiente de desarrollo usando el archivo `environment.yml`
   ```bash
   conda env create -f environment.yml
   conda activate atmoscol2023
   ```
1. Vaya a la carpeta `notebooks` y comience una sesión de `Jupyterlab`
   ```bash
   cd notebooks/
   jupyter lab
   ```
