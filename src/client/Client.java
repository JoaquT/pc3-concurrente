package client;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.Socket;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.Date;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.LineBorder;

public class Client extends JFrame {
    private String serverHost;
    private int serverPort;
    private static final String CLIENT_CHAT_ID = "0812";

    private JPanel chatMessagesPanel;
    private JScrollPane chatScrollPane;
    private JLabel lblStatus;
    private JLabel lblImageThumbnail;
    private JButton btnSelectImage;
    private JButton btnSend;
    private JTextField txtHost;
    private JTextField txtPort;
    private JLabel lblConnectionIndicator;
    private File selectedFile = null;

    public Client() {
        this("localhost", 5000);
    }

    public Client(String host, int port) {
        this.serverHost = (host != null && !host.isEmpty()) ? host : "localhost";
        this.serverPort = (port > 0) ? port : 5000;

        setTitle("Cliente - Chat Clasificador");
        setSize(780, 720);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout());

        // Cabecera superior
        JPanel topContainer = new JPanel(new BorderLayout());

        JPanel headerPanel = new JPanel(new BorderLayout());
        headerPanel.setBackground(new Color(24, 43, 73));
        headerPanel.setBorder(new EmptyBorder(12, 18, 12, 18));

        JLabel lblTitle = new JLabel("Título de diálogo: Cliente");
        lblTitle.setFont(new Font("SansSerif", Font.BOLD, 18));
        lblTitle.setForeground(Color.WHITE);

        JPanel userPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        userPanel.setOpaque(false);

        JLabel lblDot = new JLabel("●");
        lblDot.setForeground(new Color(76, 175, 80));
        lblDot.setFont(new Font("SansSerif", Font.BOLD, 16));

        JLabel lblUser = new JLabel("Usuario Validado  |  idClientChat " + CLIENT_CHAT_ID);
        lblUser.setFont(new Font("SansSerif", Font.PLAIN, 14));
        lblUser.setForeground(new Color(220, 237, 200));

        userPanel.add(lblDot);
        userPanel.add(lblUser);

        headerPanel.add(lblTitle, BorderLayout.WEST);
        headerPanel.add(userPanel, BorderLayout.EAST);
        topContainer.add(headerPanel, BorderLayout.NORTH);

        // Barra de conexión al Clúster / Red
        JPanel connPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 6));
        connPanel.setBackground(new Color(238, 242, 246));
        connPanel.setBorder(BorderFactory.createMatteBorder(0, 0, 1, 0, new Color(207, 216, 220)));

        JLabel lblHost = new JLabel("IP Servidor:");
        lblHost.setFont(new Font("SansSerif", Font.BOLD, 12));
        lblHost.setForeground(Color.BLACK);
        txtHost = new JTextField(this.serverHost, 13);
        txtHost.setFont(new Font("Monospaced", Font.PLAIN, 12));

        JLabel lblPort = new JLabel("Puerto:");
        lblPort.setFont(new Font("SansSerif", Font.BOLD, 12));
        lblPort.setForeground(Color.BLACK);
        txtPort = new JTextField(String.valueOf(this.serverPort), 5);
        txtPort.setFont(new Font("Monospaced", Font.PLAIN, 12));

        JButton btnTestConn = new JButton("Probar Conexión");
        btnTestConn.setFont(new Font("SansSerif", Font.PLAIN, 12));
        btnTestConn.setForeground(Color.BLACK);
        btnTestConn.setFocusPainted(false);
        btnTestConn.addActionListener(e -> testServerConnection());

        lblConnectionIndicator = new JLabel("● Estado de Red");
        lblConnectionIndicator.setFont(new Font("SansSerif", Font.BOLD, 12));
        lblConnectionIndicator.setForeground(Color.GRAY);

        connPanel.add(lblHost);
        connPanel.add(txtHost);
        connPanel.add(lblPort);
        connPanel.add(txtPort);
        connPanel.add(btnTestConn);
        connPanel.add(lblConnectionIndicator);
        topContainer.add(connPanel, BorderLayout.SOUTH);

        add(topContainer, BorderLayout.NORTH);

        // Panel de mensajes
        chatMessagesPanel = new JPanel();
        chatMessagesPanel.setLayout(new BoxLayout(chatMessagesPanel, BoxLayout.Y_AXIS));
        chatMessagesPanel.setBackground(new Color(245, 247, 250));
        chatMessagesPanel.setBorder(new EmptyBorder(15, 20, 15, 20));

        chatScrollPane = new JScrollPane(chatMessagesPanel);
        chatScrollPane.setBorder(null);
        chatScrollPane.getVerticalScrollBar().setUnitIncrement(16);
        add(chatScrollPane, BorderLayout.CENTER);

        addBotMessage("¡Hola! Selecciona una imagen y pulsa <b>'Identificar'</b> para clasificarla en el clúster.");

        // Panel inferior
        JPanel bottomContainer = new JPanel(new BorderLayout());
        bottomContainer.setBackground(Color.WHITE);
        bottomContainer.setBorder(BorderFactory.createMatteBorder(1, 0, 0, 0, new Color(220, 224, 230)));

        JPanel actionPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 15, 10));
        actionPanel.setOpaque(false);

        btnSelectImage = new JButton("Seleccionar Imagen");
        btnSelectImage.setFont(new Font("SansSerif", Font.PLAIN, 13));
        btnSelectImage.setFocusPainted(false);

        lblImageThumbnail = new JLabel("Ninguna imagen seleccionada");
        lblImageThumbnail.setFont(new Font("SansSerif", Font.ITALIC, 12));
        lblImageThumbnail.setForeground(Color.GRAY);

        btnSend = new JButton("Yo: Identificar");
        btnSend.setFont(new Font("SansSerif", Font.BOLD, 13));
        btnSend.setBackground(new Color(225, 230, 240));
        btnSend.setForeground(Color.BLACK);
        btnSend.setFocusPainted(false);
        btnSend.setEnabled(false);

        actionPanel.add(btnSelectImage);
        actionPanel.add(lblImageThumbnail);
        actionPanel.add(btnSend);

        JPanel statusPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 15, 5));
        statusPanel.setBackground(new Color(240, 242, 245));
        lblStatus = new JLabel("Estado: Servidor configurado en " + this.serverHost + ":" + this.serverPort);
        lblStatus.setFont(new Font("SansSerif", Font.PLAIN, 11));
        lblStatus.setForeground(new Color(100, 110, 120));
        statusPanel.add(lblStatus);

        bottomContainer.add(actionPanel, BorderLayout.CENTER);
        bottomContainer.add(statusPanel, BorderLayout.SOUTH);
        add(bottomContainer, BorderLayout.SOUTH);

        btnSelectImage.addActionListener(e -> selectImage());
        btnSend.addActionListener(e -> sendImageForIdentification());
    }

    private void selectImage() {
        JFileChooser chooser = new JFileChooser(new File("dataset"));
        int res = chooser.showOpenDialog(this);
        if (res == JFileChooser.APPROVE_OPTION) {
            selectedFile = chooser.getSelectedFile();
            lblImageThumbnail.setText(selectedFile.getName());
            lblImageThumbnail.setIcon(createThumbnail(selectedFile, 32, 32));
            lblImageThumbnail.setFont(new Font("SansSerif", Font.BOLD, 12));
            lblImageThumbnail.setForeground(new Color(33, 33, 33));
            btnSend.setEnabled(true);
        }
    }

    private ImageIcon createThumbnail(File file, int w, int h) {
        try {
            BufferedImage img = ImageIO.read(file);
            if (img != null) {
                Image scaled = img.getScaledInstance(w, h, Image.SCALE_SMOOTH);
                return new ImageIcon(scaled);
            }
        } catch (Exception ignored) {}
        return null;
    }

    private void addUserMessage(String text, File imageFile) {
        JPanel msgPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 5, 5));
        msgPanel.setOpaque(false);

        JPanel bubble = new JPanel(new BorderLayout(8, 5));
        bubble.setBackground(new Color(227, 242, 253));
        bubble.setBorder(BorderFactory.createCompoundBorder(
                new LineBorder(new Color(187, 222, 251), 1, true),
                new EmptyBorder(8, 12, 8, 12)
        ));

        JLabel lblUserText = new JLabel("<html><b>Yo: " + text + "</b></html>");
        lblUserText.setFont(new Font("SansSerif", Font.BOLD, 14));
        lblUserText.setForeground(Color.BLACK);

        bubble.add(lblUserText, BorderLayout.NORTH);

        if (imageFile != null) {
            ImageIcon preview = createThumbnail(imageFile, 120, 120);
            if (preview != null) {
                JLabel lblImg = new JLabel(preview);
                lblImg.setBorder(new LineBorder(Color.WHITE, 2));
                bubble.add(lblImg, BorderLayout.CENTER);
            }
        }

        JLabel lblTime = new JLabel(new SimpleDateFormat("HH:mm:ss").format(new Date()));
        lblTime.setFont(new Font("SansSerif", Font.PLAIN, 10));
        lblTime.setForeground(Color.GRAY);
        bubble.add(lblTime, BorderLayout.SOUTH);

        msgPanel.add(bubble);
        chatMessagesPanel.add(msgPanel);
        chatMessagesPanel.add(Box.createVerticalStrut(8));
        chatMessagesPanel.revalidate();
        scrollChatToBottom();
    }

    private void addBotMessage(String htmlContent) {
        JPanel msgPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 5));
        msgPanel.setOpaque(false);

        JPanel bubble = new JPanel(new BorderLayout(5, 5));
        bubble.setBackground(Color.WHITE);
        bubble.setBorder(BorderFactory.createCompoundBorder(
                new LineBorder(new Color(220, 224, 230), 1, true),
                new EmptyBorder(10, 14, 10, 14)
        ));

        JLabel lblBot = new JLabel("<html><body style='width: 380px;'>" + htmlContent + "</body></html>");
        lblBot.setFont(new Font("SansSerif", Font.PLAIN, 13));
        lblBot.setForeground(new Color(33, 33, 33));

        JLabel lblTime = new JLabel("Bot • " + new SimpleDateFormat("HH:mm:ss").format(new Date()));
        lblTime.setFont(new Font("SansSerif", Font.PLAIN, 10));
        lblTime.setForeground(Color.GRAY);

        bubble.add(lblBot, BorderLayout.CENTER);
        bubble.add(lblTime, BorderLayout.SOUTH);

        msgPanel.add(bubble);
        chatMessagesPanel.add(msgPanel);
        chatMessagesPanel.add(Box.createVerticalStrut(8));
        chatMessagesPanel.revalidate();
        scrollChatToBottom();
    }

    private void scrollChatToBottom() {
        SwingUtilities.invokeLater(() -> {
            JScrollBar vertical = chatScrollPane.getVerticalScrollBar();
            vertical.setValue(vertical.getMaximum());
        });
    }

    private void testServerConnection() {
        String host = (txtHost != null && !txtHost.getText().trim().isEmpty()) ? txtHost.getText().trim() : serverHost;
        int port;
        try {
            port = Integer.parseInt(txtPort.getText().trim());
        } catch (NumberFormatException ex) {
            lblConnectionIndicator.setText("✗ Puerto inválido");
            lblConnectionIndicator.setForeground(new Color(198, 40, 40));
            return;
        }

        lblConnectionIndicator.setText("● Conectando...");
        lblConnectionIndicator.setForeground(Color.DARK_GRAY);

        new SwingWorker<Boolean, Void>() {
            @Override
            protected Boolean doInBackground() {
                try (Socket s = new Socket()) {
                    s.connect(new java.net.InetSocketAddress(host, port), 2000);
                    return true;
                } catch (Exception ex) {
                    return false;
                }
            }

            @Override
            protected void done() {
                try {
                    boolean ok = get();
                    if (ok) {
                        lblConnectionIndicator.setText("✓ Conectado (" + host + ":" + port + ")");
                        lblConnectionIndicator.setForeground(new Color(46, 125, 50));
                        lblStatus.setText("Estado: Conexión exitosa con el servidor central.");
                    } else {
                        lblConnectionIndicator.setText("✗ Servidor no responde");
                        lblConnectionIndicator.setForeground(new Color(198, 40, 40));
                        lblStatus.setText("Estado: No se pudo conectar con " + host + ":" + port);
                    }
                } catch (Exception ignored) {}
            }
        }.execute();
    }

    private void sendImageForIdentification() {
        if (selectedFile == null || !selectedFile.exists()) return;

        final File imgFile = selectedFile;
        addUserMessage("Identificar", imgFile);
        btnSend.setEnabled(false);
        btnSelectImage.setEnabled(false);
        lblStatus.setText("Estado: Enviando imagen al clúster para procesamiento distribuido...");

        final String targetHost = (txtHost != null && !txtHost.getText().trim().isEmpty()) ? txtHost.getText().trim() : serverHost;
        int parsedPort;
        try {
            parsedPort = (txtPort != null) ? Integer.parseInt(txtPort.getText().trim()) : serverPort;
        } catch (NumberFormatException e) {
            parsedPort = serverPort;
        }
        final int targetPort = parsedPort;

        SwingWorker<String, Void> worker = new SwingWorker<>() {
            @Override
            protected String doInBackground() throws Exception {
                try (Socket socket = new Socket(targetHost, targetPort);
                     DataOutputStream out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
                     DataInputStream in = new DataInputStream(new BufferedInputStream(socket.getInputStream()))) {

                    byte[] bytes = Files.readAllBytes(imgFile.toPath());
                    out.writeInt(bytes.length);
                    out.write(bytes);
                    out.flush();

                    return in.readUTF();
                }
            }

            @Override
            protected void done() {
                btnSend.setEnabled(true);
                btnSelectImage.setEnabled(true);
                try {
                    String response = get();
                    lblStatus.setText("Estado: Inferencia distribuida completada con éxito.");

                    String[] parts = response.split("\\|");
                    if (parts.length >= 6) {
                        String clase = parts[0];
                        String conf = parts[1];
                        String nodeInfo = parts[2];
                        String convMs = parts[3];
                        String mlpMs = parts[4];
                        String totalMs = parts[5];

                        String botHtml = String.format(
                                "<b style='font-size:15px; color:#2E7D32;'>Bot: %s</b> " +
                                "<span style='color:#555;'>(%s%%)</span><br><br>" +
                                "<div style='font-size:12px; color:#222; background:#F8FAFC; border:1px solid #E2E8F0; padding:6px 10px; border-radius:4px; line-height:1.4;'>" +
                                "%s" +
                                "<div style='margin-top:4px; color:#64748B;'>Tiempo total: %s ms</div>" +
                                "</div>",
                                clase.toLowerCase(), conf, nodeInfo, totalMs
                        );
                        addBotMessage(botHtml);
                    } else if (parts.length >= 5) {
                        String clase = parts[0];
                        String conf = parts[1];
                        String convInfo = parts[2];
                        String mlpInfo = parts[3];
                        String totalMs = parts[4];

                        String botHtml = String.format(
                                "<b style='font-size:15px; color:#2E7D32;'>Bot: %s</b> " +
                                "<span style='color:#555;'>(%s%% de confianza)</span><br><br>" +
                                "<div style='font-size:11px; color:#333; background:#F5F5F5; padding:6px; border-radius:4px;'>" +
                                "• <b>Convolución:</b> %s<br>" +
                                "• <b>MLP:</b> %s<br>" +
                                "• <b>Tiempo Total:</b> %s ms" +
                                "</div>",
                                clase.toLowerCase(), conf, convInfo, mlpInfo, totalMs
                        );
                        addBotMessage(botHtml);
                    } else {
                        addBotMessage("<b>Bot:</b> " + response);
                    }

                } catch (Exception ex) {
                    lblStatus.setText("Error: " + ex.getMessage());
                    addBotMessage("<b style='color:#C62828;'>Error en el Clúster:</b> " + ex.getMessage());
                }
            }
        };
        worker.execute();
    }

    public static void main(String[] args) {
        String host = args.length > 0 ? args[0] : "localhost";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 5000;
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception ignored) {}
            new Client(host, port).setVisible(true);
        });
    }
}
