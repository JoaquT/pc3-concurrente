package utils;

import java.io.*;
import java.util.Random;

public class SimpleNeuralNetwork implements Serializable {
    private static final long serialVersionUID = 3L;
    public static final int MAGIC_HEADER = 0x43554245; // "CUBE"

    public int inputSize;
    public int hidden1Size;
    public int hidden2Size;
    public int outputSize;
    public double learningRate;

    public double[][] W1; public double[] b1;
    public double[][] W2; public double[] b2;
    public double[][] W3; public double[] b3;

    // Momentos de primer y segundo orden para optimizador adaptativo AdamW
    private double[][] mW1, vW1; private double[] mb1, vb1;
    private double[][] mW2, vW2; private double[] mb2, vb2;
    private double[][] mW3, vW3; private double[] mb3, vb3;
    private int adamStep = 0;

    public void initOptimizer() {
        mW1 = new double[hidden1Size][inputSize]; vW1 = new double[hidden1Size][inputSize];
        mb1 = new double[hidden1Size]; vb1 = new double[hidden1Size];
        mW2 = new double[hidden2Size][hidden1Size]; vW2 = new double[hidden2Size][hidden1Size];
        mb2 = new double[hidden2Size]; vb2 = new double[hidden2Size];
        mW3 = new double[outputSize][hidden2Size]; vW3 = new double[outputSize][hidden2Size];
        mb3 = new double[outputSize]; vb3 = new double[outputSize];
        adamStep = 0;
    }

    public static class Gradients implements Serializable {
        private static final long serialVersionUID = 3L;
        public double[][] dW1; public double[] db1;
        public double[][] dW2; public double[] db2;
        public double[][] dW3; public double[] db3;
        public int sampleCount;

        public Gradients(int in, int h1, int h2, int out) {
            dW1 = new double[h1][in]; db1 = new double[h1];
            dW2 = new double[h2][h1]; db2 = new double[h2];
            dW3 = new double[out][h2]; db3 = new double[out];
            sampleCount = 0;
        }

        public synchronized void accumulate(Gradients other) {
            for (int i = 0; i < db3.length; i++) {
                db3[i] += other.db3[i];
                for (int j = 0; j < dW3[i].length; j++) dW3[i][j] += other.dW3[i][j];
            }
            for (int i = 0; i < db2.length; i++) {
                db2[i] += other.db2[i];
                for (int j = 0; j < dW2[i].length; j++) dW2[i][j] += other.dW2[i][j];
            }
            for (int i = 0; i < db1.length; i++) {
                db1[i] += other.db1[i];
                for (int j = 0; j < dW1[i].length; j++) dW1[i][j] += other.dW1[i][j];
            }
            sampleCount += other.sampleCount;
        }
    }

    public SimpleNeuralNetwork(int inputSize, int hidden1Size, int hidden2Size, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hidden1Size = hidden1Size;
        this.hidden2Size = hidden2Size;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        W1 = new double[hidden1Size][inputSize]; b1 = new double[hidden1Size];
        W2 = new double[hidden2Size][hidden1Size]; b2 = new double[hidden2Size];
        W3 = new double[outputSize][hidden2Size]; b3 = new double[outputSize];

        Random rand = new Random(42);
        // Inicialización He / Xavier
        double std1 = Math.sqrt(2.0 / inputSize);
        for (int i = 0; i < hidden1Size; i++)
            for (int j = 0; j < inputSize; j++) W1[i][j] = rand.nextGaussian() * std1;

        double std2 = Math.sqrt(2.0 / hidden1Size);
        for (int i = 0; i < hidden2Size; i++)
            for (int j = 0; j < hidden1Size; j++) W2[i][j] = rand.nextGaussian() * std2;

        double std3 = Math.sqrt(2.0 / (hidden2Size + outputSize));
        for (int i = 0; i < outputSize; i++)
            for (int j = 0; j < hidden2Size; j++) W3[i][j] = rand.nextGaussian() * std3;

        initOptimizer();
    }

    private double relu(double x) { return Math.max(0.0, x); }
    private double reluDeriv(double x) { return x > 0.0 ? 1.0 : 0.0; }

    public double[] softmax(double[] z) {
        double max = Double.NEGATIVE_INFINITY;
        for (double val : z) if (val > max) max = val;
        double sum = 0.0;
        double[] exp = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            exp[i] = Math.exp(z[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < z.length; i++) exp[i] /= (sum > 0 ? sum : 1.0);
        return exp;
    }

    public double[] predict(double[] x) {
        double[] a1 = new double[hidden1Size];
        java.util.stream.IntStream.range(0, hidden1Size).parallel().forEach(i -> {
            double sum = b1[i];
            for (int j = 0; j < inputSize; j++) sum += W1[i][j] * x[j];
            a1[i] = relu(sum);
        });

        double[] a2 = new double[hidden2Size];
        java.util.stream.IntStream.range(0, hidden2Size).parallel().forEach(i -> {
            double sum = b2[i];
            for (int j = 0; j < hidden1Size; j++) sum += W2[i][j] * a1[j];
            a2[i] = relu(sum);
        });

        double[] z3 = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = b3[i];
            for (int j = 0; j < hidden2Size; j++) sum += W3[i][j] * a2[j];
            z3[i] = sum;
        }
        return softmax(z3);
    }

    public void computeSampleGradients(double[] x, int label, Gradients grad) {
        double[] z1 = new double[hidden1Size], a1 = new double[hidden1Size];
        for (int i = 0; i < hidden1Size; i++) {
            double sum = b1[i];
            for (int j = 0; j < inputSize; j++) sum += W1[i][j] * x[j];
            z1[i] = sum; a1[i] = relu(sum);
        }

        double[] z2 = new double[hidden2Size], a2 = new double[hidden2Size];
        for (int i = 0; i < hidden2Size; i++) {
            double sum = b2[i];
            for (int j = 0; j < hidden1Size; j++) sum += W2[i][j] * a1[j];
            z2[i] = sum; a2[i] = relu(sum);
        }

        double[] z3 = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = b3[i];
            for (int j = 0; j < hidden2Size; j++) sum += W3[i][j] * a2[j];
            z3[i] = sum;
        }
        double[] probs = softmax(z3);

        double[] dZ3 = new double[outputSize];
        for (int i = 0; i < outputSize; i++) dZ3[i] = probs[i] - (i == label ? 1.0 : 0.0);

        double[] dZ2 = new double[hidden2Size];
        for (int j = 0; j < hidden2Size; j++) {
            double err = 0;
            for (int i = 0; i < outputSize; i++) err += dZ3[i] * W3[i][j];
            dZ2[j] = err * reluDeriv(z2[j]);
        }

        double[] dZ1 = new double[hidden1Size];
        for (int j = 0; j < hidden1Size; j++) {
            double err = 0;
            for (int i = 0; i < hidden2Size; i++) err += dZ2[i] * W2[i][j];
            dZ1[j] = err * reluDeriv(z1[j]);
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hidden2Size; j++) grad.dW3[i][j] += dZ3[i] * a2[j];
            grad.db3[i] += dZ3[i];
        }
        for (int i = 0; i < hidden2Size; i++) {
            for (int j = 0; j < hidden1Size; j++) grad.dW2[i][j] += dZ2[i] * a1[j];
            grad.db2[i] += dZ2[i];
        }
        for (int i = 0; i < hidden1Size; i++) {
            for (int j = 0; j < inputSize; j++) grad.dW1[i][j] += dZ1[i] * x[j];
            grad.db1[i] += dZ1[i];
        }
        grad.sampleCount++;
    }

    public synchronized void applyGradients(Gradients grad, double lr) {
        if (grad.sampleCount == 0) return;
        if (mW1 == null) initOptimizer();

        adamStep++;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        double weightDecay = 3e-5; // Regularización L2 calibrada para balancear generalización y alta discriminación

        double b1Correction = 1.0 - Math.pow(beta1, adamStep);
        double b2Correction = 1.0 - Math.pow(beta2, adamStep);
        double invCount = 1.0 / grad.sampleCount;

        // Capa 3 (Salida)
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hidden2Size; j++) {
                double g = grad.dW3[i][j] * invCount + weightDecay * W3[i][j];
                mW3[i][j] = beta1 * mW3[i][j] + (1.0 - beta1) * g;
                vW3[i][j] = beta2 * vW3[i][j] + (1.0 - beta2) * g * g;
                double mHat = mW3[i][j] / b1Correction;
                double vHat = vW3[i][j] / b2Correction;
                W3[i][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
            }
            double gb = grad.db3[i] * invCount;
            mb3[i] = beta1 * mb3[i] + (1.0 - beta1) * gb;
            vb3[i] = beta2 * vb3[i] + (1.0 - beta2) * gb * gb;
            double mHatB = mb3[i] / b1Correction;
            double vHatB = vb3[i] / b2Correction;
            b3[i] -= lr * mHatB / (Math.sqrt(vHatB) + eps);
        }

        // Capa 2 (Oculta 2)
        for (int i = 0; i < hidden2Size; i++) {
            for (int j = 0; j < hidden1Size; j++) {
                double g = grad.dW2[i][j] * invCount + weightDecay * W2[i][j];
                mW2[i][j] = beta1 * mW2[i][j] + (1.0 - beta1) * g;
                vW2[i][j] = beta2 * vW2[i][j] + (1.0 - beta2) * g * g;
                double mHat = mW2[i][j] / b1Correction;
                double vHat = vW2[i][j] / b2Correction;
                W2[i][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
            }
            double gb = grad.db2[i] * invCount;
            mb2[i] = beta1 * mb2[i] + (1.0 - beta1) * gb;
            vb2[i] = beta2 * vb2[i] + (1.0 - beta2) * gb * gb;
            double mHatB = mb2[i] / b1Correction;
            double vHatB = vb2[i] / b2Correction;
            b2[i] -= lr * mHatB / (Math.sqrt(vHatB) + eps);
        }

        // Capa 1 (Entrada -> Oculta 1)
        for (int i = 0; i < hidden1Size; i++) {
            for (int j = 0; j < inputSize; j++) {
                double g = grad.dW1[i][j] * invCount + weightDecay * W1[i][j];
                mW1[i][j] = beta1 * mW1[i][j] + (1.0 - beta1) * g;
                vW1[i][j] = beta2 * vW1[i][j] + (1.0 - beta2) * g * g;
                double mHat = mW1[i][j] / b1Correction;
                double vHat = vW1[i][j] / b2Correction;
                W1[i][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
            }
            double gb = grad.db1[i] * invCount;
            mb1[i] = beta1 * mb1[i] + (1.0 - beta1) * gb;
            vb1[i] = beta2 * vb1[i] + (1.0 - beta2) * gb * gb;
            double mHatB = mb1[i] / b1Correction;
            double vHatB = vb1[i] / b2Correction;
            b1[i] -= lr * mHatB / (Math.sqrt(vHatB) + eps);
        }
    }

    public void trainStep(double[] x, int label) {
        Gradients g = new Gradients(inputSize, hidden1Size, hidden2Size, outputSize);
        computeSampleGradients(x, label, g);
        applyGradients(g, learningRate);
    }

    // Serialización binaria

    public void writeBinary(DataOutputStream out) throws IOException {
        out.writeInt(MAGIC_HEADER);
        out.writeInt(inputSize);
        out.writeInt(hidden1Size);
        out.writeInt(hidden2Size);
        out.writeInt(outputSize);
        out.writeDouble(learningRate);

        for (int i = 0; i < hidden1Size; i++)
            for (int j = 0; j < inputSize; j++) out.writeDouble(W1[i][j]);
        for (int i = 0; i < hidden1Size; i++) out.writeDouble(b1[i]);

        for (int i = 0; i < hidden2Size; i++)
            for (int j = 0; j < hidden1Size; j++) out.writeDouble(W2[i][j]);
        for (int i = 0; i < hidden2Size; i++) out.writeDouble(b2[i]);

        for (int i = 0; i < outputSize; i++)
            for (int j = 0; j < hidden2Size; j++) out.writeDouble(W3[i][j]);
        for (int i = 0; i < outputSize; i++) out.writeDouble(b3[i]);
        out.flush();
    }

    public static SimpleNeuralNetwork readBinary(DataInputStream in) throws IOException {
        int magic = in.readInt();
        if (magic != MAGIC_HEADER) {
            throw new IOException("Formato de modelo inválido: magic mismatch");
        }
        int inSize = in.readInt();
        int h1 = in.readInt();
        int h2 = in.readInt();
        int out = in.readInt();
        double lr = in.readDouble();

        SimpleNeuralNetwork net = new SimpleNeuralNetwork(inSize, h1, h2, out, lr);
        for (int i = 0; i < h1; i++)
            for (int j = 0; j < inSize; j++) net.W1[i][j] = in.readDouble();
        for (int i = 0; i < h1; i++) net.b1[i] = in.readDouble();

        for (int i = 0; i < h2; i++)
            for (int j = 0; j < h1; j++) net.W2[i][j] = in.readDouble();
        for (int i = 0; i < h2; i++) net.b2[i] = in.readDouble();

        for (int i = 0; i < out; i++)
            for (int j = 0; j < h2; j++) net.W3[i][j] = in.readDouble();
        for (int i = 0; i < out; i++) net.b3[i] = in.readDouble();

        return net;
    }

    public void save(String filepath) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(filepath);
             BufferedOutputStream bos = new BufferedOutputStream(fos);
             DataOutputStream dos = new DataOutputStream(bos)) {
            writeBinary(dos);
        }
    }

    public static SimpleNeuralNetwork load(String filepath) throws IOException {
        File file = new File(filepath);
        if (!file.exists()) {
            throw new FileNotFoundException("Archivo de modelo no encontrado: " + filepath);
        }
        try (FileInputStream fis = new FileInputStream(file);
             BufferedInputStream bis = new BufferedInputStream(fis);
             DataInputStream dis = new DataInputStream(bis)) {
            dis.mark(8);
            int magic = dis.readInt();
            dis.reset();
            if (magic == MAGIC_HEADER) {
                return readBinary(dis);
            }
        }
        // Fallback a ObjectInputStream para compatibilidad con modelos antiguos serializados
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            return (SimpleNeuralNetwork) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IOException("Clase no encontrada al deserializar: " + e.getMessage());
        }
    }
}