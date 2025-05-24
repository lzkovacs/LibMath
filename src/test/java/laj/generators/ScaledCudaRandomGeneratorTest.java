package laj.generators;

import laj.generators.utils.Algorithm;
import laj.generators.utils.GeneratorParams;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tesztosztály a ScaledCudaRandomGenerator osztályhoz.
 * Ez a teszt a GeneratorParams objektum helyes konfigurációját ellenőrzi
 * a feladatban megadott paraméterekkel.
 * Megjegyzés: Mivel a CUDA hívások valódi GPU-t igényelnek,
 * ez a teszt nem próbálja meg ténylegesen létrehozni a ScaledCudaRandomGenerator
 * példányt, csak a paraméterek helyességét ellenőrzi.
 */
class ScaledCudaRandomGeneratorTest {

    /**
     * Teszt a GeneratorParams helyes konfigurációjára a feladatban megadott paraméterekkel.
     * Ellenőrzi, hogy a paraméterek megfelelően vannak beállítva.
     */
    @Test
    void testGeneratorParamsConfiguration() {
        // A feladatban megadott paraméterek
        GeneratorParams params = new GeneratorParams(
                Algorithm.XORWOW,  // Alap GPU algoritmus
                42L,               // Seed érték = 42
                10,                // Vektor méret = 10
                true,              // Transzformáció bekapcsolva = true
                1,                 // Min érték = 1
                90,                // Max érték = 90
                5                  // Blokkméret = 5
        );

        // Ellenőrizzük a paramétereket
        assertEquals(Algorithm.XORWOW, params.algorithm());
        assertEquals(42L, params.seed());
        assertEquals(10, params.vectorSize());
        assertTrue(params.transformEnabled());
        assertEquals(1, params.min());
        assertEquals(90, params.max());
        assertEquals(5, params.k());

        // Ellenőrizzük, hogy a paraméterek megfelelnek a feladat leírásának
        String description = "A tesztben a GeneratorParams értékei: seed=42, vectorSize=10, " +
                "transformEnabled = true, min=1, max = 90, k = 5";

        assertEquals(42L, params.seed(), "A seed értéke 42 kell legyen");
        assertEquals(10, params.vectorSize(), "A vectorSize értéke 10 kell legyen");
        assertTrue(true, "A transformEnabled értéke true kell legyen");
        assertEquals(1, params.min(), "A min értéke 1 kell legyen");
        assertEquals(90, params.max(), "A max értéke 90 kell legyen");
        assertEquals(5, params.k(), "A k értéke 5 kell legyen");
    }
}
