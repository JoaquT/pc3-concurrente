package node_java;

import java.io.*;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import utils.ImageClassifierHelper;
import utils.SimpleNeuralNetwork;

public class JavaNode {
    private static final String SERVER_HOST = "localhost";
    private static final int SERVER_PORT = 5001;
    private static final String[] CLASSES = {"gato", "mariposa", "perro"};

    private final String nodeId;
    private final String host;
    private final int port;
    private SimpleNeuralNetwork model;

    public JavaNode(String nodeId, String host, int port) {
        this.nodeId = nodeId;
        this.host = host;
        this.port = port;

        File modelFile = new File("modelo_animales.bin");
        if (modelFile.exists()) {
            try {
                this.model = SimpleNeuralNetwork.load(modelFile.getPath());
                System.out.println("[" + nodeId + "] 'modelo_animales.bin' cargado con éxito.");
            } catch (Exception e) {
                System.err.println("[" + nodeId + "] Aviso al cargar modelo: " + e.getMessage());
                this.model = new SimpleNeuralNetwork(ImageClassifierHelper.TOTAL_FEATURES, 128, 64, 3, 0.03);
            }
        } else {
            this.model = new SimpleNeuralNetwork(ImageClassifierHelper.TOTAL_FEATURES, 128, 64, 3, 0.03);
        }
    }

    public void start() {
        System.out.println("[" + nodeId + "] Conectando al Servidor Central en " + host + ":" + port + "...");

        while (true) {
            try (Socket socket = new Socket(host, port);
                 DataInputStream in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
                 DataOutputStream out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()))) {

                System.out.println("[" + nodeId + "] ¡Conectado al Clúster!");
                out.writeUTF("REGISTER:" + nodeId + ":JAVA:CONV,MLP");
                out.flush();

                while (true) {
                    String cmd;
                    try {
                        cmd = in.readUTF();
                    } catch (EOFException e) {
                        System.out.println("[" + nodeId + "] Servidor desconectado.");
                        break;
                    }

                    if (cmd.equals("QUIT")) break;

                    if (cmd.equals("TASK_CLASSIFY")) {
                        int grayLen = in.readInt();
                        double[] flatGray = new double[grayLen];
                        for (int i = 0; i < grayLen; i++) flatGray[i] = in.readDouble();

                        int histLen = in.readInt();
                        double[] colorHist = new double[histLen];
                        for (int i = 0; i < histLen; i++) colorHist[i] = in.readDouble();

                        long t0 = System.nanoTime();
                        double[][] gray2D = new double[32][32];
                        for (int y = 0; y < 32; y++) {
                            System.arraycopy(flatGray, y * 32, gray2D[y], 0, 32);
                        }
                        double[] convFeat = ImageClassifierHelper.computeConvolutions(gray2D);
                        double[] fullFeat = new double[histLen + convFeat.length];
                        System.arraycopy(colorHist, 0, fullFeat, 0, histLen);
                        System.arraycopy(convFeat, 0, fullFeat, histLen, convFeat.length);
                        double convMs = (System.nanoTime() - t0) / 1_000_000.0;

                        long t1 = System.nanoTime();
                        double[] probs = model.predict(fullFeat);
                        int best = 0;
                        for (int i = 1; i < probs.length; i++) {
                            if (probs[i] > probs[best]) best = i;
                        }
                        double mlpMs = (System.nanoTime() - t1) / 1_000_000.0;

                        String bestName = CLASSES[best].toUpperCase();
                        double conf = probs[best] * 100.0;

                        out.writeUTF("RESULT_CLASSIFY:" + nodeId + ":" + bestName + ":" + String.format(java.util.Locale.US, "%.2f", conf) + ":" + String.format(java.util.Locale.US, "%.2f", convMs) + ":" + String.format(java.util.Locale.US, "%.2f", mlpMs));
                        out.writeInt(probs.length);
                        for (double p : probs) out.writeDouble(p);
                        out.flush();

                    } else if (cmd.equals("TASK_CONV")) {
                        // Micro-chunk 1: Convolución
                        int grayLen = in.readInt();
                        double[] flatGray = new double[grayLen];
                        for (int i = 0; i < grayLen; i++) flatGray[i] = in.readDouble();

                        int histLen = in.readInt();
                        double[] colorHist = new double[histLen];
                        for (int i = 0; i < histLen; i++) colorHist[i] = in.readDouble();

                        long t0 = System.nanoTime();
                        double[][] gray2D = new double[32][32];
                        for (int y = 0; y < 32; y++) {
                            System.arraycopy(flatGray, y * 32, gray2D[y], 0, 32);
                        }
                        double[] convFeat = ImageClassifierHelper.computeConvolutions(gray2D);

                        double[] fullFeat = new double[histLen + convFeat.length];
                        System.arraycopy(colorHist, 0, fullFeat, 0, histLen);
                        System.arraycopy(convFeat, 0, fullFeat, histLen, convFeat.length);
                        double elapsedMs = (System.nanoTime() - t0) / 1_000_000.0;

                        out.writeUTF("RESULT_CONV:" + nodeId + ":" + String.format("%.2f", elapsedMs));
                        out.writeInt(fullFeat.length);
                        for (double v : fullFeat) out.writeDouble(v);
                        out.flush();

                    } else if (cmd.equals("TASK_MLP")) {
                        // Micro-chunk 2: MLP Inferencia
                        int featLen = in.readInt();
                        double[] feat = new double[featLen];
                        for (int i = 0; i < featLen; i++) feat[i] = in.readDouble();

                        long t0 = System.nanoTime();
                        double[] probs = model.predict(feat);
                        int best = 0;
                        for (int i = 1; i < probs.length; i++) {
                            if (probs[i] > probs[best]) best = i;
                        }
                        double elapsedMs = (System.nanoTime() - t0) / 1_000_000.0;

                        String bestName = CLASSES[best].toUpperCase();
                        double conf = probs[best] * 100.0;

                        out.writeUTF("RESULT_MLP:" + nodeId + ":" + bestName + ":" + String.format("%.2f", conf) + ":" + String.format("%.2f", elapsedMs));
                        out.writeInt(probs.length);
                        for (double p : probs) out.writeDouble(p);
                        out.flush();

                    } else if (cmd.equals("TASK_TRAIN_BATCH")) {
                        // Entrenamiento Distribuido: Cómputo de Gradientes locales
                        int batchSize = in.readInt();
                        SimpleNeuralNetwork.Gradients grads = new SimpleNeuralNetwork.Gradients(
                                model.inputSize, model.hidden1Size, model.hidden2Size, model.outputSize
                        );

                        long t0 = System.nanoTime();
                        for (int b = 0; b < batchSize; b++) {
                            int label = in.readInt();
                            int featLen = in.readInt();
                            double[] x = new double[featLen];
                            for (int i = 0; i < featLen; i++) x[i] = in.readDouble();
                            model.computeSampleGradients(x, label, grads);
                        }
                        double elapsedMs = (System.nanoTime() - t0) / 1_000_000.0;

                        out.writeUTF("RESULT_TRAIN_BATCH:" + nodeId + ":" + batchSize + ":" + String.format("%.2f", elapsedMs));
                        out.writeInt(grads.sampleCount);

                        // Enviar dW1, db1
                        for (int i = 0; i < model.hidden1Size; i++)
                            for (int j = 0; j < model.inputSize; j++) out.writeDouble(grads.dW1[i][j]);
                        for (int i = 0; i < model.hidden1Size; i++) out.writeDouble(grads.db1[i]);

                        // Enviar dW2, db2
                        for (int i = 0; i < model.hidden2Size; i++)
                            for (int j = 0; j < model.hidden1Size; j++) out.writeDouble(grads.dW2[i][j]);
                        for (int i = 0; i < model.hidden2Size; i++) out.writeDouble(grads.db2[i]);

                        // Enviar dW3, db3
                        for (int i = 0; i < model.outputSize; i++)
                            for (int j = 0; j < model.hidden2Size; j++) out.writeDouble(grads.dW3[i][j]);
                        for (int i = 0; i < model.outputSize; i++) out.writeDouble(grads.db3[i]);
                        out.flush();

                    } else if (cmd.equals("SYNC_WEIGHTS")) {
                        int inSize = in.readInt();
                        int h1 = in.readInt();
                        int h2 = in.readInt();
                        int outSize = in.readInt();

                        model = new SimpleNeuralNetwork(inSize, h1, h2, outSize, 0.03);
                        for (int i = 0; i < h1; i++)
                            for (int j = 0; j < inSize; j++) model.W1[i][j] = in.readDouble();
                        for (int i = 0; i < h1; i++) model.b1[i] = in.readDouble();

                        for (int i = 0; i < h2; i++)
                            for (int j = 0; j < h1; j++) model.W2[i][j] = in.readDouble();
                        for (int i = 0; i < h2; i++) model.b2[i] = in.readDouble();

                        for (int i = 0; i < outSize; i++)
                            for (int j = 0; j < h2; j++) model.W3[i][j] = in.readDouble();
                        for (int i = 0; i < outSize; i++) model.b3[i] = in.readDouble();

                        out.writeUTF("SYNC_OK");
                        out.flush();
                        System.out.println("[" + nodeId + "] Pesos del clúster sincronizados exitosamente.");

                    } else if (cmd.equals("PING")) {
                        out.writeUTF("PONG");
                        out.flush();
                    }
                }
            } catch (Exception e) {
                System.err.println("[" + nodeId + "] Error de red: " + e.getMessage() + ". Reintentando en 3s...");
                try { Thread.sleep(3000); } catch (InterruptedException ignored) {}
            }
        }
    }

    public static void main(String[] args) {
        String nodeId = args.length > 0 ? args[0] : "Nodo_Java_1";
        String host = args.length > 1 ? args[1] : SERVER_HOST;
        int port = args.length > 2 ? Integer.parseInt(args[2]) : SERVER_PORT;

        JavaNode node = new JavaNode(nodeId, host, port);
        node.start();
    }
}
