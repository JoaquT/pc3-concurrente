package client;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.Base64;
import java.util.Date;

/**
 * Open AI Cubes - Cliente Desktop Multiplataforma (Java Swing)
 *
 * Cumple con las Restricciones Estrictas:
 * - CERO WebSockets / CERO librerías externas de red.
 * - Comunicación mediante Sockets TCP nativos (java.net.Socket) hacia el puerto 5000 del Backend Central.
 * - Permite enviar mensajes de chat y seleccionar imágenes locales, codificarlas en Base64
 *   (java.util.Base64) y mostrar el dictamen de la CNN en el flujo del chat.
 */
public class DesktopChatClient extends JFrame {

    private final String serverHost;
    private final int serverPort;
    private final String username;

    private Socket tcpSocket;
    private PrintWriter socketOut;
    private BufferedReader socketIn;

    private JTextArea chatArea;
    private JTextField messageField;
    private JLabel statusLabel;
    private JLabel imagePreviewLabel;
    private JButton sendTextBtn;
    private JButton selectImageBtn;

    public DesktopChatClient(String host, int port, String username) {
        super("Open AI Cubes - Desktop TCP Client (" + username + ")");
        this.serverHost = host;
        this.serverPort = port;
        this.username = username;

        initUI();
        connectToBackendTcp();
    }

    private void initUI() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(760, 560);
        setLocationRelativeTo(null);

        JPanel root = new JPanel(new BorderLayout(10, 10));
        root.setBorder(new EmptyBorder(12, 12, 12, 12));
        root.setBackground(new Color(18, 24, 38));

        // Cabecera de estado
        JPanel headerPanel = new JPanel(new BorderLayout());
        headerPanel.setBackground(new Color(30, 41, 59));
        headerPanel.setBorder(new EmptyBorder(8, 12, 8, 12));

        JLabel titleLabel = new JLabel("Open AI Cubes — Chat & Clasificador CNN (TCP Socket Nativo :5000)");
        titleLabel.setForeground(Color.WHITE);
        titleLabel.setFont(new Font("SansSerif", Font.BOLD, 14));

        statusLabel = new JLabel("Desconectado");
        statusLabel.setForeground(new Color(248, 113, 113));
        statusLabel.setFont(new Font("SansSerif", Font.BOLD, 12));

        headerPanel.add(titleLabel, BorderLayout.WEST);
        headerPanel.add(statusLabel, BorderLayout.EAST);
        root.add(headerPanel, BorderLayout.NORTH);

        // Área central: Historial de Chat + Panel lateral de Previsualización
        chatArea = new JTextArea();
        chatArea.setEditable(false);
        chatArea.setLineWrap(true);
        chatArea.setWrapStyleWord(true);
        chatArea.setFont(new Font("Monospaced", Font.PLAIN, 13));
        chatArea.setBackground(new Color(15, 23, 42));
        chatArea.setForeground(new Color(226, 232, 240));
        chatArea.setMargin(new Insets(10, 10, 10, 10));

        JScrollPane scrollPane = new JScrollPane(chatArea);
        scrollPane.setBorder(BorderFactory.createLineBorder(new Color(51, 65, 85)));

        JPanel previewPanel = new JPanel(new BorderLayout(6, 6));
        previewPanel.setPreferredSize(new Dimension(200, 0));
        previewPanel.setBackground(new Color(30, 41, 59));
        previewPanel.setBorder(new EmptyBorder(10, 10, 10, 10));

        JLabel previewTitle = new JLabel("Última Imagen Enviada", SwingConstants.CENTER);
        previewTitle.setForeground(new Color(148, 163, 184));
        previewTitle.setFont(new Font("SansSerif", Font.BOLD, 12));

        imagePreviewLabel = new JLabel("Sin imagen", SwingConstants.CENTER);
        imagePreviewLabel.setForeground(new Color(100, 116, 139));
        imagePreviewLabel.setBorder(BorderFactory.createDashedBorder(new Color(71, 85, 105)));

        selectImageBtn = new JButton("Enviar Imagen (Base64)");
        selectImageBtn.setBackground(new Color(16, 185, 129));
        selectImageBtn.setForeground(Color.WHITE);
        selectImageBtn.setFocusPainted(false);
        selectImageBtn.setFont(new Font("SansSerif", Font.BOLD, 12));
        selectImageBtn.addActionListener(e -> chooseAndSendImage());

        previewPanel.add(previewTitle, BorderLayout.NORTH);
        previewPanel.add(imagePreviewLabel, BorderLayout.CENTER);
        previewPanel.add(selectImageBtn, BorderLayout.SOUTH);

        root.add(scrollPane, BorderLayout.CENTER);
        root.add(previewPanel, BorderLayout.EAST);

        // Barra inferior: Entrada de mensajes de Chat
        JPanel bottomPanel = new JPanel(new BorderLayout(8, 0));
        bottomPanel.setOpaque(false);

        messageField = new JTextField();
        messageField.setFont(new Font("SansSerif", Font.PLAIN, 14));
        messageField.addActionListener(e -> sendChatMessage());

        sendTextBtn = new JButton("Enviar Chat");
        sendTextBtn.setBackground(new Color(59, 130, 246));
        sendTextBtn.setForeground(Color.WHITE);
        sendTextBtn.setFocusPainted(false);
        sendTextBtn.setFont(new Font("SansSerif", Font.BOLD, 13));
        sendTextBtn.addActionListener(e -> sendChatMessage());

        bottomPanel.add(messageField, BorderLayout.CENTER);
        bottomPanel.add(sendTextBtn, BorderLayout.EAST);
        root.add(bottomPanel, BorderLayout.SOUTH);

        setContentPane(root);
    }

    private void connectToBackendTcp() {
        Thread connector = new Thread(() -> {
            try {
                appendChat("[Sistema] Conectando mediante Socket TCP puro a " + serverHost + ":" + serverPort + "...");
                tcpSocket = new Socket(serverHost, serverPort);
                socketOut = new PrintWriter(tcpSocket.getOutputStream(), true, StandardCharsets.UTF_8);
                socketIn = new BufferedReader(new InputStreamReader(tcpSocket.getInputStream(), StandardCharsets.UTF_8));

                SwingUtilities.invokeLater(() -> {
                    statusLabel.setText("Conectado (" + serverHost + ":" + serverPort + ")");
                    statusLabel.setForeground(new Color(74, 222, 128));
                });

                // Hilo lector continuo del Socket TCP
                String serverLine;
                while ((serverLine = socketIn.readLine()) != null) {
                    handleServerMessage(serverLine);
                }
            } catch (IOException ex) {
                appendChat("[Error TCP] No se pudo mantener la conexión con el Backend Central: " + ex.getMessage());
                SwingUtilities.invokeLater(() -> {
                    statusLabel.setText("Desconectado");
                    statusLabel.setForeground(new Color(248, 113, 113));
                });
            }
        }, "Desktop-TCP-Listener");
        connector.setDaemon(true);
        connector.start();
    }

    private void handleServerMessage(String line) {
        if (line.startsWith("WELCOME:")) {
            appendChat("[Backend Central] " + line.substring(8));
        } else if (line.startsWith("CHAT:")) {
            appendChat(line.substring(5));
        } else if (line.startsWith("AI_RESULT:")) {
            String json = line.substring(10);
            appendChat("--------------------------------------------------\n"
                    + "[Respuesta CNN vía RabbitMQ]\n"
                    + json + "\n"
                    + "--------------------------------------------------");
        } else if (line.startsWith("AI_ERROR:")) {
            appendChat("[Error Clasificación IA] " + line.substring(9));
        } else {
            appendChat("[Servidor] " + line);
        }
    }

    private void sendChatMessage() {
        String text = messageField.getText().trim();
        if (text.isEmpty() || socketOut == null) return;

        socketOut.println("CHAT:(" + username + ") " + text);
        messageField.setText("");
    }

    private void chooseAndSendImage() {
        if (socketOut == null) {
            JOptionPane.showMessageDialog(this, "No hay conexión TCP activa con el Backend Central.", "Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        JFileChooser chooser = new JFileChooser(new File("."));
        chooser.setDialogTitle("Seleccionar imagen para clasificar en el Nodo IA");
        chooser.setFileFilter(new FileNameExtensionFilter("Imágenes (JPG, PNG, JPEG, BMP)", "jpg", "jpeg", "png", "bmp"));

        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File selectedFile = chooser.getSelectedFile();
            // Previsualización en el cliente Swing
            ImageIcon icon = new ImageIcon(selectedFile.getAbsolutePath());
            Image scaled = icon.getImage().getScaledInstance(160, 160, Image.SCALE_SMOOTH);
            imagePreviewLabel.setText("");
            imagePreviewLabel.setIcon(new ImageIcon(scaled));

            // Codificar en Base64 y enviar por Socket TCP en hilo separado para no bloquear la UI
            new Thread(() -> {
                try {
                    byte[] fileBytes = Files.readAllBytes(selectedFile.toPath());
                    String base64Image = Base64.getEncoder().encodeToString(fileBytes);

                    appendChat("[" + username + "] Enviando imagen '" + selectedFile.getName()
                            + "' (" + fileBytes.length + " bytes -> Base64) por Socket TCP...");

                    // Notificar al chat y enviar comando IMG:<base64> al puerto 5000
                    socketOut.println("CHAT:(" + username + ") subió la imagen '" + selectedFile.getName() + "' para clasificación CNN.");
                    socketOut.println("IMG:" + base64Image);
                } catch (IOException ex) {
                    appendChat("[Error] No se pudo leer o codificar la imagen: " + ex.getMessage());
                }
            }, "Image-Base64-Sender").start();
        }
    }

    private void appendChat(String msg) {
        String time = new SimpleDateFormat("HH:mm:ss").format(new Date());
        SwingUtilities.invokeLater(() -> {
            chatArea.append("[" + time + "] " + msg + "\n");
            chatArea.setCaretPosition(chatArea.getDocument().getLength());
        });
    }

    public static void main(String[] args) {
        String host = args.length > 0 ? args[0] : "localhost";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 5000;
        String user = args.length > 2 ? args[2] : "DesktopUser_" + (System.currentTimeMillis() % 1000);

        SwingUtilities.invokeLater(() -> {
            DesktopChatClient client = new DesktopChatClient(host, port, user);
            client.setVisible(true);
        });
    }
}
