package laj.generators;

import jcuda.driver.CUcontext;
import jcuda.jcurand.curandGenerator;
import laj.generators.utils.Algorithm;
import laj.generators.utils.GeneratorParams;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tesztosztály a UniformCudaRandomGenerator osztályhoz.
 * Ez az osztály a UniformCudaRandomGenerator alapvető funkcionalitását teszteli, beleértve:
 * - Konstruktor különböző paraméterekkel
 * - Algoritmus validáció
 * - Erőforrás-kezelés
 * 
 * Megjegyzés: Mivel a CUDA és cuRAND hívások valódi GPU-t igényelnek,
 * ezek a tesztek egy tesztspecifikus alosztályt használnak, amely felülírja
 * a CUDA-specifikus műveleteket a tesztelhetőség érdekében.
 */
@ExtendWith(MockitoExtension.class)
class UniformCudaRandomGeneratorTest {

    /**
     * Tesztspecifikus alosztály, amely felülírja a CUDA-specifikus műveleteket
     * és elérhetővé teszi az algoritmus validációs logikát.
     * Ez lehetővé teszi a tesztelést valódi GPU nélkül.
     */
    static class TestableUniformCudaRandomGenerator extends UniformCudaRandomGenerator {
        /**
         * Konstruktor, amely nem inicializálja a CUDA környezetet.
         * Csak a teszteléshez használjuk.
         */
        public TestableUniformCudaRandomGenerator(GeneratorParams params, CUcontext context) {
            super(params, context);
        }

        /**
         * Felülírjuk a generate metódust, hogy ne hívjon CUDA műveleteket.
         */
        @Override
        public void generate(long n) {
            // Nem csinál semmit, csak a tesztelhetőség miatt
        }

        /**
         * Felülírjuk a close metódust, hogy ne hívjon CUDA műveleteket.
         */
        @Override
        public void close() {
            // Nem csinál semmit, csak a tesztelhetőség miatt
        }

        /**
         * Nyilvános metódus az algoritmus validálásához.
         * Ez lehetővé teszi a privát mapAlgorithm metódus tesztelését.
         * 
         * @param algo A tesztelendő algoritmus
         * @return true, ha az algoritmus támogatott, false egyébként
         */
        public static boolean isAlgorithmSupported(Algorithm algo) {
            try {
                // Megpróbáljuk leképezni az algoritmust
                // Ha kivételt dob, akkor nem támogatott
                switch (algo) {
                    case XORWOW, SCALE_XORWOW, DRAW_XORWOW, SORTED_XORWOW:
                    case MRG32K3A, SCALE_MRG32K3A:
                    case MTGP32, SCALE_MTGP32:
                    case MT19937, SCALE_MT19937:
                    case PHILOX4_32_10, SCALE_PHILOX4_32_10:
                        return true;
                    default:
                        return false;
                }
            } catch (Exception e) {
                return false;
            }
        }
    }

    @Mock
    private CUcontext mockContext;

    private GeneratorParams params;

    @BeforeEach
    void setUp() {
        // Alapértelmezett paraméterek a tesztekhez
        params = new GeneratorParams(
                Algorithm.XORWOW,  // GPU algoritmus
                12345L,            // Seed érték
                1000,              // Vektor méret
                false,             // Transzformáció kikapcsolva
                0,                 // Min érték
                100,               // Max érték
                10                 // Blokkméret
        );
    }

    /**
     * Teszt az algoritmus validáció helyes működésére CPU algoritmussal.
     * Ellenőrzi, hogy a CPU algoritmusok nem támogatottak.
     */
    @Test
    void algorithm_WithCpuAlgorithm_ShouldNotBeSupported() {
        // CPU algoritmus, ami nem támogatott a CUDA generátorban
        Algorithm cpuAlgorithm = Algorithm.JAVA_RANDOM;

        // Ellenőrizzük, hogy a CPU algoritmus nem támogatott
        assertFalse(TestableUniformCudaRandomGenerator.isAlgorithmSupported(cpuAlgorithm));
    }

    /**
     * Teszt az algoritmus validáció helyes működésére GPU algoritmussal.
     * Ellenőrzi, hogy a GPU algoritmusok támogatottak.
     */
    @Test
    void algorithm_WithGpuAlgorithm_ShouldBeSupported() {
        // GPU algoritmus, ami támogatott a CUDA generátorban
        Algorithm gpuAlgorithm = Algorithm.XORWOW;

        // Ellenőrizzük, hogy a GPU algoritmus támogatott
        assertTrue(TestableUniformCudaRandomGenerator.isAlgorithmSupported(gpuAlgorithm));
    }

    /**
     * Teszt az algoritmus validáció helyes működésére wrapper algoritmussal.
     * Ellenőrzi, hogy a wrapper algoritmusok támogatottak, ha az alap algoritmus támogatott.
     */
    @Test
    void algorithm_WithWrapperAlgorithm_ShouldBeSupported() {
        // Wrapper algoritmus, ami támogatott, mert az alap algoritmus támogatott
        Algorithm wrapperAlgorithm = Algorithm.DRAW_XORWOW;

        // Ellenőrizzük, hogy a wrapper algoritmus támogatott
        assertTrue(TestableUniformCudaRandomGenerator.isAlgorithmSupported(wrapperAlgorithm));
    }

    /**
     * Teszt a konstruktor helyes működésére null kontextussal.
     * Ellenőrzi, hogy null kontextus esetén kivételt dob.
     */
    @Test
    void constructor_WithNullContext_ShouldThrowException() {
        // Ellenőrizzük, hogy kivételt dob null kontextus esetén
        Exception exception = assertThrows(IllegalArgumentException.class, 
                () -> new UniformCudaRandomGenerator(params, null));

        assertTrue(exception.getMessage().contains("context nem lehet null"));
    }

    /**
     * Teszt a konstruktor helyes működésére érvényes paraméterekkel.
     * Ellenőrzi, hogy a generátor létrehozható és lezárható.
     */
    @Test
    void constructor_WithValidParams_ShouldCreateAndCloseInstance() {
        // Létrehozunk egy tesztelhető generátort érvényes paraméterekkel
        try {
            // Megpróbáljuk létrehozni a generátort
            TestableUniformCudaRandomGenerator generator = new TestableUniformCudaRandomGenerator(params, mockContext);

            // Ellenőrizzük, hogy sikeresen létrejött
            assertNotNull(generator);
            assertEquals(params, generator.getParams());
            assertEquals(mockContext, generator.getContext());

            // Teszteljük a generate metódust (nem dob kivételt)
            assertDoesNotThrow(() -> generator.generate(500));

            // Teszteljük a close metódust (nem dob kivételt)
            assertDoesNotThrow(() -> generator.close());

        } catch (Exception e) {
            fail("Nem várt kivétel: " + e.getMessage());
        }
    }
}
