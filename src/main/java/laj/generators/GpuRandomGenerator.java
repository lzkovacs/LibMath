package laj.generators;

import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdeviceptr;
import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import laj.generators.utils.GeneratorParams;
import lombok.Getter;

/**
 * <h2>GpuRandomGenerator</h2>
 * <p>
 * Ez az absztrakt osztály egy GPU-alapú véletlenszám-generátor közös alapjait tartalmazza.
 * Fő célja, hogy biztosítsa az alapvető erőforrás-kezelést (memória allokáció, kontextus kezelés)
 * és a paraméterezhetőséget különböző véletlenszám-generátor implementációk számára.
 * </p>
 *
 * <h3>Fő tulajdonságai:</h3>
 * <ul>
 *     <li><strong>context</strong>: a CUDA kontextus, amelyben a generátor fut</li>
 *     <li><strong>devPtr</strong>: az eszköz (GPU) memóriacímét leíró objektum</li>
 *     <li><strong>nBytes</strong>: a GPU-n lefoglalt memória mérete byte-ban</li>
 *     <li><strong>bOwnCOntext</strong>: jelzi, hogy az osztály tulajdonolja-e a kontextust</li>
 *     <li><strong>params</strong>: a generátor futásához szükséges konfigurációs paraméterek</li>
 * </ul>
 *
 * <p>Minden leszármazott osztálynak implementálnia kell a {@link #generate(long)} és
 * a {@link #close()} metódusokat.</p>
 */
@Getter
public abstract class GpuRandomGenerator {
    /**
     * A CUDA kontextus, amelyben a generátor működik.
     */
    protected final CUcontext context;

    /**
     * Az eszköz (GPU) memória területének mutatója,
     * ahová az eredményeket mentjük.
     */
    protected final CUdeviceptr devPtr;

    /**
     * A GPU memóriában lefoglalt adatsor mérete byte-ban.
     */
    protected final long nBytes;

    /**
     * Jelzi, hogy az objektum a kontextus tulajdonosa-e (saját maga kezeli-e a létrehozást/lezárást).
     */
    protected final boolean bOwnCOntext;

    /**
     * A generátor működéséhez szükséges paramétereket leíró objektum.
     */
    protected final GeneratorParams params;

    /**
     * Konstruktor, amely inicializálja az osztály mezőit és lefoglalja a szükséges GPU memóriát.
     * Mielőtt memóriát foglalnánk, beállítjuk az aktuális CUDA kontextust.
     *
     * @param params A generátor működéséhez szükséges {@link GeneratorParams} példány.
     * @param context A CUDA kontextus, amelyben a generátor működik (nem lehet {@code null}).
     * @throws IllegalArgumentException ha a context értéke {@code null}.
     */
    GpuRandomGenerator(GeneratorParams params, CUcontext context){
        if (context == null) {
            throw new IllegalArgumentException("GpuRandomGenerator: A CUDA context nem lehet null");
        }
        this.bOwnCOntext = false;
        this.context = context;
        this.params = params;
        // A tárolandó vektor méretének megfelelő memóriaallokáció a GPU-n
        this.nBytes = (long) params.vectorSize() * Sizeof.FLOAT;
        this.devPtr = new CUdeviceptr();

        // Az adott CUDA kontextus beállítása és memória allokációja
        cuCtxSetCurrent(context);
        cuMemAlloc(devPtr, nBytes);
    }

    /**
     * Véletlenszám-generálás: minden leszármazottnak kötelező implementálnia.
     *
     * @param n Az előállítandó véletlen számok száma.
     */
    public abstract void generate(long n);

    /**
     * Erőforrás-felszabadítás. Az absztrakt metódust minden leszármazottnak implementálnia kell,
     * aki felszabadítja a GPU erőforrásokat.
     */
    public abstract void close();
}