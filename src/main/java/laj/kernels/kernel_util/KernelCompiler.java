/*
 * Copyright (c) 2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */

package laj.kernels.kernel_util;

public interface KernelCompiler {
    /**
     * Fordítja a resource .cu fájlt PTX formátumra.
     * @param resourceName A forrás resource neve (pl. "kernel.gpu_utils/transform.cu")
     * @return A lefordított .ptx fájl elérési útja
     * Fontos, hogy fix, előre lefordított PTX fájlok beemelésére is alkalmassá tesz
     */
    String compile(String resourceName);
}