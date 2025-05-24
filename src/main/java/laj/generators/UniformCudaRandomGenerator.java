package laj.generators;

import laj.generators.utils.Algorithm;
import laj.generators.utils.GeneratorParams;
import jcuda.driver.CUcontext;
import jcuda.driver.JCudaDriver;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.*;

/**
 * Egyenletes (uniform) eloszlású véletlenszám-generátor GPU-n, CUDA és cuRAND használatával.
 * <p>
 * A generátor a cuRAND könyvtárat használja különböző algoritmusok támogatásával, 
 * és a véletlen vektorokat közvetlenül a GPU memóriájába (device memory) generálja.
 * </p>
 * 
 * <h2>Példahasználat:</h2>
 * <pre>{@code
 * try (UniformCudaRandomGenerator generator = new UniformCudaRandomGenerator(params, context)) {
 *     generator.generate(n); // n darab véletlen számot generál
 *     // Véletlen vektor elérése pl. generator.devPtr
 * }
 * }</pre>
 * 
 * <p>
 * Erőforrás-menedzsment: 
 * A generátor használata után mindig hívja meg a {@link #close()} metódust, akár automatizáltan 
 * a try-with-resources szerkezetben, hogy a GPU erőforrásokat felszabadítsa!
 * </p>
 *
 * @see GpuRandomGenerator Szülőosztály, amely általános GPU-s generálást támogat
 */
public class UniformCudaRandomGenerator extends GpuRandomGenerator {
    
    /**
     * A cuRAND véletlenszám-generátor CUDA-n.
     */
    private final curandGenerator generator;
    
    /**
     * Létrehozza és inicializálja a GPU-s véletlenszám-generátort a megadott kontextussal.
     *
     * @param params A véletlenszám-generálás paraméterei (vektorméret, algoritmus, seed)
     * @param providedContext A használandó CUDA kontextus
     * @throws RuntimeException ha bármely CUDA/cuRAND hívás sikertelen
     */
    public UniformCudaRandomGenerator(GeneratorParams params, CUcontext providedContext) {
        super(params, providedContext);
        JCudaDriver.setExceptionsEnabled(true);
        JCurand.setExceptionsEnabled(true);

        this.generator = new curandGenerator();
        
        curandCreateGenerator(generator, mapAlgorithm(params.algorithm()));
        curandSetPseudoRandomGeneratorSeed(generator, params.seed());
    }

    /**
     * Egy új uniform eloszlású véletlen vektor generálása a GPU memóriába (deviceVector-ba).
     * A számok 0 és 1 közötti (lebegőpontos, float) értékek, uniform eloszlásban.
     * Az alkalmazás végén mindig hívandó a {@link #close()}, hogy a GPU erőforrásokat felszabadítsa.
     *
     * @param n Generálandó elemek száma (float véletlenszámok száma)
     * @throws RuntimeException ha a CUDA/cuRAND hívás hibába ütközik
     */
    @Override
    public void generate(long n) {
        // Beállítjuk az aktuális kontextust a generálás előtt
        cuCtxSetCurrent(context);
        curandGenerateUniform(generator, devPtr, n);
        // Beállítjuk az aktuális kontextust a szinkronizálás előtt
        cuCtxSetCurrent(context);
        cuCtxSynchronize();
    }
    
    /**
     * Az összes GPU erőforrás, vektor és generátor felszabadítása, kontextus lezárása.
     * <br>Mindig hívja meg, ha már nincs szükség a generátorra. (Segíthet a try-with-resources.)
     * A hívás után a példány nem használható tovább.
     */
    @Override
    public void close() {
        // Beállítjuk az aktuális kontextust a generátor felszabadítása előtt
        cuCtxSetCurrent(context);
        curandDestroyGenerator(generator);
        // Beállítjuk az aktuális kontextust a memória felszabadítása előtt
        cuCtxSetCurrent(context);
        cuMemFree(devPtr);

        // Csak akkor semmisítjük meg a kontextust, ha mi hoztuk létre
        if (bOwnCOntext) {
            cuCtxDestroy(context);
        }
    }

    /**
     * Leképezi az absztrakt {@link Algorithm} algoritmusokat a cuRAND natív RNG típus konstansaira.
     *
     * @param algo a választott algoritmus ENUM értéke
     * @return a cuRAND RNG típuskód (például {@link jcuda.jcurand.curandRngType#CURAND_RNG_PSEUDO_XORWOW})
     * @throws IllegalArgumentException ha a megadott algoritmus nincs támogatva
     */
    private static int mapAlgorithm(Algorithm algo) {
        return switch (algo) {
            case XORWOW, SCALE_XORWOW, DRAW_XORWOW, SORTED_XORWOW -> CURAND_RNG_PSEUDO_XORWOW;
            case MRG32K3A, SCALE_MRG32K3A -> CURAND_RNG_PSEUDO_MRG32K3A;
            case MTGP32, SCALE_MTGP32 -> CURAND_RNG_PSEUDO_MTGP32;
            case MT19937, SCALE_MT19937 -> CURAND_RNG_PSEUDO_MT19937;
            case PHILOX4_32_10, SCALE_PHILOX4_32_10 -> CURAND_RNG_PSEUDO_PHILOX4_32_10;
            default -> throw new IllegalArgumentException(
                    "GpuRandomGenerator - Nem támogatott GPU algoritmus: " + algo
            );
        };
    }
}