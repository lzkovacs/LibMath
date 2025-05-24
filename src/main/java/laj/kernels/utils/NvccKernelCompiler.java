/*
 * Copyright (c) 2024-2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */
package laj.kernels.utils;

import jcuda.CudaException;
import jcuda.driver.CUdevice;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;
import lombok.extern.log4j.Log4j2;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static jcuda.driver.CUdevice_attribute.*;
import static jcuda.driver.JCudaDriver.*;

/**
 * KernelCompiler implementáció NVCC használatával.
 */
@Log4j2
public class NvccKernelCompiler implements KernelCompiler {
    private static final int IO_BUFFER_SIZE = 8 * 1024;

    static {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
    }

    @Override
    public String compile(String resourceName) {
        log.debug("PTX előkészítése resource alapján: {}", resourceName);
        int device = getFirstDevice();
        String arch = "sm_" + computeComputeCapability(device);
        String[] nvccArgs = {"-arch=" + arch, "-lcurand"};
        log.debug("NVCC argumentumok: {}", Arrays.toString(nvccArgs));

        try (InputStream in = NvccKernelCompiler.class.getClassLoader().getResourceAsStream(resourceName)) {
            if (in == null) {
                throw new CudaException("Resource nem található: " + resourceName);
            }
            File cuFile = createTempFile(in, ".cu");
            String ptx = invokeNvcc(cuFile.getAbsolutePath(), "ptx", true, nvccArgs);
            log.info("PTX fájl elkészült: {}", ptx);
            return ptx;
        } catch (IOException e) {
            log.error("I/O hiba PTX előkészítése közben", e);
            throw new CudaException("I/O hiba PTX előkészítésekor", e);
        }
    }

    @SuppressWarnings("SameParameterValue")
    private File createTempFile(InputStream in, String ext) throws IOException {
        File temp = File.createTempFile("kernel_", ext);
        temp.deleteOnExit();
        try (BufferedInputStream bis = new BufferedInputStream(in, IO_BUFFER_SIZE);
             BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(temp), IO_BUFFER_SIZE)) {
            byte[] buf = new byte[IO_BUFFER_SIZE];
            int len;
            while ((len = bis.read(buf)) != -1) out.write(buf, 0, len);
        }
        log.debug("Ideiglenes CU fájl: {}", temp.getAbsolutePath());
        return temp;
    }

    private String invokeNvcc(String cuFileName, @SuppressWarnings("SameParameterValue") String targetType, @SuppressWarnings("SameParameterValue") boolean force, String... args) {
        log.debug("NVCC indítása: {} -> {}", cuFileName, targetType);
        String existing = prepareOutput(cuFileName, targetType, force);
        if (existing != null) {
            log.debug("Újrafordítás mellőzve: {}", existing);
            return existing;
        }
        String[] cmd = buildCmd(cuFileName, targetType, args);
        log.debug("NVCC parancs: {}", Arrays.toString(cmd));
        execNvcc(cmd);
        String output = cuFileName.replaceAll("\\.cu$", "." + targetType);
        log.debug("NVCC kimenet: {}", output);
        return output;
    }

    private String prepareOutput(String file, String type, boolean force) {
        String lower = type.toLowerCase();
        if (!"ptx".equals(lower) && !"cubin".equals(lower)) throw new IllegalArgumentException("Cél: ptx vagy cubin: " + type);
        String out = file.replaceAll("\\.cu$", "." + lower);
        return (new File(out).exists() && !force) ? out : null;
    }

    private String[] buildCmd(String file, String type, String... args) {
        List<String> cmd = new ArrayList<>();
        cmd.add("nvcc"); cmd.add("-m" + System.getProperty("sun.arch.data.model"));
        cmd.add("-" + type.toLowerCase()); cmd.addAll(Arrays.asList(args));
        cmd.add(file); cmd.add("-o"); cmd.add(file.replaceAll("\\.cu$", "." + type));
        return cmd.toArray(new String[0]);
    }

    private void execNvcc(String[] cmd) {
        try {
            Process p = new ProcessBuilder(cmd).start();
            String err = readErr(p);
            if (p.waitFor() != 0) throw new CudaException("NVCC sikertelen: " + err);
        } catch (Exception e) {
            Thread.currentThread().interrupt();
            log.error("NVCC hiba", e);
            throw new CudaException("NVCC hiba", e);
        }
    }

    private String readErr(Process p) throws IOException {
        try (InputStream es = p.getErrorStream(); ByteArrayOutputStream b = new ByteArrayOutputStream()) {
            byte[] buf = new byte[IO_BUFFER_SIZE]; int r;
            while ((r = es.read(buf)) != -1) b.write(buf, 0, r);
            return b.toString();
        }
    }

    private int getFirstDevice() {
        CUdevice d = new CUdevice();
        if (cuDeviceGet(d, 0) != CUresult.CUDA_SUCCESS)
            throw new CudaException("Eszköz lekérési hiba: " + CUresult.stringFor(cuDeviceGet(d, 0)));
        return 0;
    }

    private int computeComputeCapability(int id) {
        CUdevice d = new CUdevice(); cuDeviceGet(d, id);
        int[] maj = {0}, min = {0}; cuDeviceGetAttribute(maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d);
        cuDeviceGetAttribute(min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d);
        int cc = maj[0]*10 + min[0]; log.debug("Számítási képesség: {}", cc); return cc;
    }
}