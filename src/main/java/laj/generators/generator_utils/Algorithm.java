package laj.generators.generator_utils;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Véletlenszám-generáláshoz használható algoritmusok enumerációja.
 * <p>
 * A "DRAW_*", "SCALE_*" és "SORTED_*" típusok belül egy alapszintű uniform algoritmust használnak.
 * A getBase() metódus visszaadja a tényleges, alapszintű algoritmust.
 * A getType() metódussal megkülönböztethetők a CPU és GPU algoritmusok.
 * </p>
 */
@JsonFormat(shape = JsonFormat.Shape.OBJECT)
public enum Algorithm {
    // CPU–algoritmusok
    JAVA_RANDOM   (AlgorithmType.CPU, null,    "A Java beépített véletlenszám-generátora, gyors és egyszerű."),
    SFMT          (AlgorithmType.CPU, null,    "Az SFMT a Mersenne Twister gyorsabb, CPU-optimalizált változata."),
    TAUSWORTHE    (AlgorithmType.CPU, null,    "Kombinált Tausworthe eljáráson alapuló CPU-generátor."),

    // GPU–algoritmusok
    XORWOW        (AlgorithmType.GPU, null,    "Az XORWOW egy GPU-alapú XOR-wow eljárás."),
    MRG32K3A      (AlgorithmType.GPU, null,    "Az MRG32k3a algoritmus GPU-verziója."),
    MTGP32        (AlgorithmType.GPU, null,    "Mersenne Twister GPU-optimalizált változata."),
    MT19937       (AlgorithmType.GPU, null,    "Klasszikus Mersenne Twister GPU-implementáció."),
    PHILOX4_32_10 (AlgorithmType.GPU, null,    "Philox algoritmus GPU-változata."),

    // SCALE wrapper algoritmusok
    SCALE_XORWOW        (AlgorithmType.GPU, XORWOW,        "Az XORWOW skálázott változata."),
    SCALE_MRG32K3A      (AlgorithmType.GPU, MRG32K3A,      "Az MRG32K3A skálázott változata."),
    SCALE_MTGP32        (AlgorithmType.GPU, MTGP32,        "Az MTGP32 skálázott változata."),
    SCALE_MT19937       (AlgorithmType.GPU, MT19937,       "Az MT19937 skálázott változata."),
    SCALE_PHILOX4_32_10 (AlgorithmType.GPU, PHILOX4_32_10, "A PHILOX4_32_10 skálázott változata."),

    // DRAW wrapper algoritmusok
    DRAW_XORWOW     (AlgorithmType.GPU, XORWOW,      "Véletlenszámok ismétlés nélküli húzása XORWOW-val."),
    SORTED_XORWOW   (AlgorithmType.GPU, XORWOW,      "K-elemű blokkokba rendezett véletlenszámok XORWOW-val."),
    DRAW_JAVA_RANDOM(AlgorithmType.CPU, JAVA_RANDOM,"Véletlenszámok ismétlés nélküli húzása JavaRandom-dal.");

    /**
     * -- GETTER --
     * CPU vagy GPU
     */
    @Getter
    private final AlgorithmType type;
    /**
     * -- GETTER --
     * Ha wrapper, akkor mi az alap
     */
    @Getter
    private final Algorithm    base;
    private final String       description;

    Algorithm(AlgorithmType type, Algorithm base, String description) {
        this.type        = type;
        this.base        = base;
        this.description = description;
    }

    /** Emberi magyarázat JSON-ban is */
    @JsonProperty("description")
    public String getDescription() {
        return description;
    }
    /** Wrapper algoritmusoknál true */
    public boolean isWrapper() {
        return base != null;
    }

    /** Az algoritmus neve JSON-ban */
    @JsonProperty("name")
    public String getName() {
        return name();
    }

    /** A kategória neve JSON-ban (CPU vagy GPU) */
    @JsonProperty("type")
    public String getTypeName() {
        return type.name();
    }

    /** A base algoritmus neve JSONban, null, ha nincs */
    @JsonProperty("base")
    public String getBaseName() {
        return base == null ? null : base.name();
    }

    /** Utility: az összes wrapper algoritmus, amelyek erre az alapra épülnek */
    public List<Algorithm> getWrappers() {
        return Arrays.stream(values())
                .filter(a -> this.equals(a.base))
                .collect(Collectors.toList());
    }

    /** Static: az összes CPU algoritmus */
    public static List<Algorithm> cpuAlgorithms() {
        return Arrays.stream(values())
                .filter(a -> a.type == AlgorithmType.CPU && !a.isWrapper())
                .collect(Collectors.toList());
    }

    /** Static: az összes GPU algoritmus */
    public static List<Algorithm> gpuAlgorithms() {
        return Arrays.stream(values())
                .filter(a -> a.type == AlgorithmType.GPU && !a.isWrapper())
                .collect(Collectors.toList());
    }

    /** Static: az összes wrapper algoritmus */
    public static List<Algorithm> wrapperAlgorithms() {
        return Arrays.stream(values())
                .filter(Algorithm::isWrapper)
                .collect(Collectors.toList());
    }
}