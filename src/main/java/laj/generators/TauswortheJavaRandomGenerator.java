package laj.generators;

import laj.generators.utils.GeneratorParams;

/**
 * TauswortheJavaRandomGenerator osztály
 * Ez az osztály a Tausworthe algoritmus alapján generál véletlenszámokat.
 * Három különböző állapotú (state) változót használ, amelyet egy seed-ből inicializál.
 */
public class TauswortheJavaRandomGenerator extends CpuRandomGenerator {

    // Véletlenszám-generátor belső állapotai
    private long s1, s2, s3;

    // Az előállítandó számok tartománya (max - min + 1)
    private final int range;
    // Az előállítandó számok alsó határa
    private final int min;

    /**
     * Konstruktor, mely inicializálja a generátor állapotait a paraméterek alapján
     *
     * @param params Paraméterobjektum, amely tartalmazza a minimumot, maximumot és seedet
     */
    public TauswortheJavaRandomGenerator(GeneratorParams params) {
        super(params);
        range = params.max() - params.min() + 1;
        min = params.min();
        long seed = params.seed();
        this.s1 = seed;
        this.s2 = seed + 17; // Ideálisan ide is különböző seed szükséges
        this.s3 = seed + 29; // És ide is különböző seed szükséges
    }

    /**
     * n db véletlen számot generál, és eltárolja őket a vector tömbben
     * 
     * @param n A legenerálandó számok száma
     */
    @Override
    public void generate(long n) {
        for (int i = 0; i < n; i++) {
            long raw = nextRaw(); // Nyers véletlenszám generálása
            vector[i] = (float) ((raw >>> 1) % range) + min; // Tartományba illesztés és eltárolás
        }
    }

    /**
     * Tausworthe lépés függvény - egy lépést végez az adott állapotváltozón
     * 
     * @param z aktuális állapot
     * @param s shift paraméter
     * @param q xor shift paraméter
     * @param r shift paraméter
     * @param m maszkolás paramétere
     * @return új állapotérték
     */
    private long tausworthe(long z, int s, int q, int r, long m) {
        long b = (((z << q) ^ z) >> s); // b köztes érték számítása
        return ((z & m) << r) ^ b;      // új állapot kiszámítása
    }

    /**
     * Három részállapotból egy új véletlenszámot állít elő a Tausworthe algoritmus szerint
     * 
     * @return nyers véletlenszám
     */
    private long nextRaw() {
        s1 = tausworthe(s1, 13, 19, 12, 0xFFFFFFFEL);
        s2 = tausworthe(s2, 2, 25, 4, 0xFFFFFFF8L);
        s3 = tausworthe(s3, 3, 11, 17, 0xFFFFFFF0L);
        return s1 ^ s2 ^ s3; // A három állapotból egy véletlenszámot számolunk XOR-ral
    }
}