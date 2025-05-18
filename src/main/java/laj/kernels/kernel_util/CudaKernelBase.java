/*
 * Copyright (c) 2024-2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */
package laj.kernels.kernel_util;

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
    protected final CUcontext context;
    protected CUmodule module;
    protected CUfunction function;
    protected final NvccKernelCompiler compiler;

    /**
     * @param context Létező CUcontext
     * @param compiler A NvccKernelCompiler implementáció
     */
    public CudaKernelBase(CUcontext context, NvccKernelCompiler compiler) {
        this.context = context;
        this.compiler = compiler;
        initKernel();
    }

    /**
     * Egyszerűsített konstruktor az alapértelmezett NvccKernelCompiler-rel
     */
    public CudaKernelBase(CUcontext context) {
        this(context, new NvccKernelCompiler());
    }

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

    protected abstract String getKernelFileName();
    protected abstract String getKernelFunctionName();

    @Override
    public void close() {
        if (module != null) {
            // Beállítjuk a kontextust a modul felszabadítása előtt
            cuCtxSetCurrent(context);
            cuModuleUnload(module);
            log.info("Modul felszabadítva "+getKernelFunctionName());
        }
    }
}