/*
 * Copyright (c) 2024-2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */
package laj.kernels.utils;

import jcuda.driver.CUcontext;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import lombok.extern.log4j.Log4j2;

import static jcuda.driver.JCudaDriver.*;

/**
 * Alap osztály CUDA kernel modul betöltéséhez és függvény pointer lekéréséhez.
 */
@Log4j2
public abstract class CudaKernelBase implements AutoCloseable {
    /** A használt CUDA kontextus. */
    protected final CUcontext context;
    /** A betöltött CUDA modul. */
    protected CUmodule module;
    /** A lekért CUDA kernel függvény. */
    protected CUfunction function;
    /** A kernel fordításáért felelős fordító példány. */
    protected final NvccKernelCompiler compiler;

    /**
     * Létrehoz egy új CudaKernelBase példányt a megadott kontextussal és fordítóval.
     *
     * @param context Létező CUcontext (CUDA kontextus)
     * @param compiler Az NvccKernelCompiler implementáció, amely a kernel forrás fordítását végzi
     */
    public CudaKernelBase(CUcontext context, NvccKernelCompiler compiler) {
        this.context = context;
        this.compiler = compiler;
        initKernel();
    }

    /**
     * Egyszerűsített konstruktor az alapértelmezett {@link NvccKernelCompiler}-rel.
     *
     * @param context Létező CUcontext (CUDA kontextus)
     */
    public CudaKernelBase(CUcontext context) {
        this(context, new NvccKernelCompiler());
    }

    /**
     * Inicializálja a CUDA modult és betölti a kernel függvényt.
     * Fordítja az adott kernel forrást (PTX generálás), betölti a modult,
     * ezután lekéri a megadott kernel függvény pointerét.
     * Hibás betöltés esetén kivételt dob.
     */
    private void initKernel() {
        String ptx = compiler.compile(getKernelFileName());
        module = new CUmodule();

        // Beállítjuk a kontextust a modul betöltése előtt
        cuCtxSetCurrent(context);
        int err = cuModuleLoad(module, ptx);
        if (err != CUresult.CUDA_SUCCESS) {
            log.error("PTX betöltés sikertelen: {}", CUresult.stringFor(err));
            throw new RuntimeException("PTX betöltés sikertelen: " + CUresult.stringFor(err));
        }
        log.info("Modul betöltve: {}", ptx);

        function = new CUfunction();
        // Beállítjuk a kontextust a függvény lekérése előtt
        cuCtxSetCurrent(context);
        err = cuModuleGetFunction(function, module, getKernelFunctionName());
        if (err != CUresult.CUDA_SUCCESS) {
            log.error("Kernel pointer lekérés sikertelen: {}", CUresult.stringFor(err));
            throw new RuntimeException("Kernel pointer sikertelen: " + CUresult.stringFor(err));
        }
        log.info("Függvény '{}' betöltve", getKernelFunctionName());
    }

    /**
     * Visszaadja a kernel forrásfájl nevét (például: "kernel.cu"), amelyet le kell fordítani.
     *
     * @return A kernel CUDA forrásfájl neve
     */
    protected abstract String getKernelFileName();

    /**
     * Visszaadja annak a függvénynek a nevét, amelyet a betöltött modulból le kell kérni.
     *
     * @return A kernel függvény neve a modulban
     */
    protected abstract String getKernelFunctionName();

    /**
     * Felszabadítja a betöltött CUDA modult, ha az még létezik.
     * A metódus többszöri hívása biztonságos, de csak egyszer hajtja végre a felszabadítást.
     */
    @Override
    public void close() {
        if (module != null) {
            // Beállítjuk a kontextust a modul felszabadítása előtt
            cuCtxSetCurrent(context);
            cuModuleUnload(module);
            module = null;
            log.info("Modul felszabadítva {}", getKernelFunctionName());
        }
    }
}