# PyTrad - Biblioteca de Trading en Python

## Nomenclatura de Indicadores con IndicatorName

PyTrad implementa una estructura orientada a objetos para la nomenclatura de indicadores técnicos. Esta estructura facilita la organización, manipulación y representación de los indicadores.

### Clase IndicatorName

La clase `IndicatorName` proporciona una representación estructurada de los nombres de indicadores con las siguientes propiedades:

-   **base_name**: Nombre base del indicador (ej: "MA", "RSI", "MACD")
-   **parameters**: Lista de parámetros del indicador (ej: [14] para RSI de período 14)
-   **uses_returns**: Booleano que indica si el indicador usa retornos en lugar de precios
-   **component**: Para indicadores complejos, el nombre del componente (ej: "upper_band" para Bandas de Bollinger)

### Formato de cadena

Los indicadores se representan como cadenas con el siguiente formato:

1. **Indicadores simples**: `BASE_PARAM1_PARAM2...`
   -   Ejemplo: `RSI_14` o `MA_20`

2. **Indicadores con retornos**: `BASE_PARAM1_PARAM2...-returns`
   -   Ejemplo: `RSI_14-returns`

3. **Indicadores complejos**: `BASE_PARAM1_PARAM2...-COMPONENT`
   -   Ejemplo: `BB_20_2-upper_band`

4. **Indicadores complejos con retornos**: `BASE_PARAM1_PARAM2...-returns-COMPONENT`
   -   Ejemplo: `MACD_12_26_9-returns-signal_line`

### Ventajas de este enfoque

-   **Claridad semántica**: Cada parte del nombre tiene un significado específico
-   **Facilidad de parsing**: La estructura es predecible y fácil de analizar
-   **Tipo seguro**: Al usar objetos en lugar de strings, se evitan errores tipográficos
-   **Extensibilidad**: Facilita la adición de nuevas propiedades o funcionalidades

### Ejemplo de uso

```python
# Crear un IndicatorName
rsi = IndicatorName("RSI", [14])
bb = IndicatorName("BB", [20, 2])
macd = IndicatorName("MACD", [12, 26, 9])

# Agregar componentes
bb_upper = bb.with_component("upper_band")
macd_signal = macd.with_component("signal_line")

# Convertir a string
rsi_str = rsi.to_string()  # "RSI_14"
bb_upper_str = bb_upper.to_string()  # "BB_20_2-upper_band"

# Parsear desde string
parsed_rsi = IndicatorName.from_string("RSI_14")
parsed_macd = IndicatorName.from_string("MACD_12_26_9-returns-macd_line")
```

### Integración con Window

La clase `Window` ha sido actualizada para trabajar con objetos `IndicatorName` como claves en lugar de strings. Esto proporciona una mayor seguridad de tipos y facilita el manejo de indicadores complejos.

El flujo de trabajo típico es:

1. Crear indicadores con la clase Indicator (RSI, MA, MACD, etc.)
2. Cada indicador expone una propiedad `name` que devuelve un objeto `IndicatorName`
3. Al añadir indicadores a una ventana con `add_indicator`, se utiliza este objeto como clave
4. Al convertir a DataFrame con `to_df()`, los nombres de columna se generan a partir de `IndicatorName.to_string()`

### Migración desde el formato anterior

Si tienes datos en el formato anterior (con todas las partes separadas por guiones bajos), puedes convertirlos al nuevo formato utilizando la clase `IndicatorName.from_string()` después de actualizar manualmente las cadenas.
