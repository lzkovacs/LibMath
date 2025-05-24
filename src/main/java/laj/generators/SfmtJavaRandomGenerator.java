/*
 * Copyright (c) 2024-2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */
package laj.generators;

import laj.generators.utils.GeneratorParams;
import laj.generators.utils.SFMTParam;

/**
 * SFMT-alapú véletlenszám-generátor implementációja Java nyelven.
 * A generátor a SIMD-oriented Fast Mersenne Twister algoritmus paramétereit és állapotát kezeli.
 */
public class SfmtJavaRandomGenerator extends CpuRandomGenerator {
    // Az SFMT paramétereit tároló objektum
    private final SFMTParam param;
    // Az aktuális SFMT állapot, 128-bites egységek tömbjeként
    private final W128T[] state;
    // Az aktuális index az állapot tömbben
    private int idx = 0;
    // A generált számok alsó határa (lehet negatív is)
    private final int min;
    // A generált számok értékkészletének nagysága
    private final int range;

    /**
     * Konstruktor, amely inicializálja a generátort a megadott paraméterekkel.
     * @param params A generátor paraméterei (min, max, seed)
     */
    public SfmtJavaRandomGenerator(GeneratorParams params) {
        super(params);

        min = params.min();
        range = params.max() - params.min() + 1;
        this.param = SFMTParam.P19937; // Alapértelmezett: Mersenne Twister 19937 paraméter
        this.state = initState(param.SFMT_N);
        setSeed((int) params.seed());
    }

    /**
     * A véletlenszám-vektor feltöltése n darab generált számmal.
     * @param n Hány számot generáljunk
     */
    @Override
    public void generate(long n) {
        for (int i = 0; i < n; i++) {
            int r = nextInt();
            // A generált számot az elvárt tartományba illesztjük
            vector[i] = ((r >>> 1) % range) + min;
        }
    }

    /**
     * Egy új 32 bites egész véletlenszámot szolgáltat.
     * Ha szükséges, frissíti az állapotot.
     * @return Egy 32 bites véletlenszám
     */
    private int nextInt() {
        if (idx >= param.SFMT_N32) {
            genNextState();
            idx = 0;
        }
        return getInt(idx++);
    }

    /**
     * Az SFMT algoritmus következő állapotának generálása.
     */
    private void genNextState() {
        W128T r1 = state[param.SFMT_N - 2];
        W128T r2 = state[param.SFMT_N - 1];
        // Végigmegyünk az állapoton és rekurzióval új értékeket számítunk
        for (int i = 0; i < param.SFMT_N; i++) {
            W128T out = new W128T();
            doRecursion(out,
                    state[i],
                    state[(i + param.SFMT_POS1) % param.SFMT_N],
                    r1, r2);
            r1 = r2;
            r2 = out;
            state[i] = out;
        }
    }

    /**
     * A generátor állapotának inicializálása a megadott maggal (seed).
     * @param seed Az inicializáló mag értéke
     */
    private void setSeed(int seed) {
        state[0].u(0, seed);
        for (int i = 1; i < param.SFMT_N32; i++) {
            int prev = getInt(i - 1);
            // Lineáris inicializáló formula – megegyezik a standard Mersenne Twisterrel
            setInt(i, 1812433253 * (prev ^ (prev >>> 30)) + i);
        }
        idx = param.SFMT_N32;
        periodCertification(); // Tanúsítás, hogy biztosan hosszú periódusú legyen az állapot
    }

    /**
     * Létrehozza és inicializálja az SFMT állapotot.
     * @param size A szükséges állapot hossza
     * @return W128T tömb, üresen inicializált állapottal
     */
    private static W128T[] initState(int size) {
        W128T[] s = new W128T[size];
        for (int i = 0; i < size; i++) s[i] = new W128T();
        return s;
    }

    /**
     * Adott indexű 32 bites egész kinyerése az állapotból.
     * @param i Index
     * @return 32 bites egész érték
     */
    private int getInt(int i) {
        return state[i / (W128T.BUFFER_SIZE / Integer.BYTES)].u(i % (W128T.BUFFER_SIZE / Integer.BYTES));
    }

    /**
     * Adott indexű 32 bites érték beállítása az állapotban.
     * @param i Index
     * @param v Állítani kívánt érték
     */
    private void setInt(int i, int v) {
        state[i / (W128T.BUFFER_SIZE / Integer.BYTES)].u(i % (W128T.BUFFER_SIZE / Integer.BYTES), v);
    }

    /**
     * Az SFMT magját adó rekurzív függvény.
     * Négy állapotblokk és paraméterek alapján új blokkot számol.
     */
    private void doRecursion(W128T r, W128T a, W128T b, W128T c, W128T d) {
        W128T x = a.lshift128(param.SFMT_SL2);
        W128T y = c.rshift128(param.SFMT_SR2);
        for (int j = 0; j < 4; j++) {
            r.u(j,
                a.u(j)
                ^ x.u(j)
                ^ ((b.u(j) >>> param.SFMT_SR1) & param.SFMT_MSK(j))
                ^ y.u(j)
                ^ (d.u(j) << param.SFMT_SL1)
            );
        }
    }

    /**
     * Az állapot tanúsítása, hogy teljesüljenek a véletlenszám-generálás periódusának feltételei,
     * különösen, hogy a periodicitás paramétere ne legyen nulla.
     */
    private void periodCertification() {
        int inner = 0;
        for (int i = 0; i < 4; i++) {
            inner ^= state[0].u(i) & param.SFMT_PARITY(i);
        }
        // Bitenkénti redukció
        for (int shift = 16; shift > 0; shift >>>= 1) inner ^= inner >>> shift;
        inner &= 1;
        if (inner == 1) return; // Már megfelelő az állapot
        // Egy bitet átbillentünk, hogy biztosítsuk a periodicitást
        for (int i = 0; i < 4; i++) {
            int work = 1;
            for (int j = 0; j < 32; j++) {
                if ((work & param.SFMT_PARITY(i)) != 0) {
                    state[0].u(i, state[0].u(i) ^ work);
                    return;
                }
                work <<= 1;
            }
        }
    }

    /**
     * Belső 128-bites típus. Négy 32 bites egészből áll, shift műveletekkel és indexelt hozzáféréssel.
     */
    private static class W128T {
        // 128 bit = 16 bájt
        static final int BUFFER_SIZE = 128 / 8;
        // Négy egészként tárolva
        private final int[] data = new int[BUFFER_SIZE / Integer.BYTES];

        /**
         * Egy adott 32 bites egész lekérése az index alapján.
         */
        int u(int idx) { return data[idx]; }

        /**
         * Egy adott 32 bites érték beállítása az indexre.
         */
        void u(int idx, int v) { data[idx] = v; }

        /**
         * Biteltolás balra (128 biten belül).
         * Implementálandó: 128-bites balra shift léptetés.
         */
        W128T lshift128(int s) { /* implementálandó */ return this; }

        /**
         * Biteltolás jobbra (128 biten belül).
         * Implementálandó: 128-bites jobbra shift léptetés.
         */
        W128T rshift128(int s) { /* implementálandó */ return this; }
    }
}