import cv2
import numpy as np
import random
import time
from flask import Flask, render_template, Response

app = Flask(__name__)

def detectar_cor(frame, lower_color, upper_color):
    """Detecta a cor especificada no frame e retorna uma máscara."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.GaussianBlur(mask, (5, 5), 100)

def gerar_posicao_aleatoria(frame_width, frame_height, tamanho_alvo, distancia_minima, posicao_atual):
    """Gera uma nova posição aleatória para o balão, respeitando a distância mínima."""
    for _ in range(10):  # Limite de 10 tentativas
        margem_lateral = 20
        x = random.randint(margem_lateral, frame_width - tamanho_alvo - margem_lateral)
        y = random.randint(50, frame_height - tamanho_alvo - 50)

        # Verifica a distância mínima da nova posição em relação à posição atual
        distancia = np.sqrt((x - posicao_atual[0]) ** 2 + (y - posicao_atual[1]) ** 2)
        if distancia >= distancia_minima:
            return (x, y)
    
    return posicao_atual  # Retorna a posição atual se não encontrar uma nova válida

def detectar_colisao(x, y, w, h, alvo_x, alvo_y, alvo_tamanho):
    """Verifica se houve colisão entre o objeto e o balão."""
    return (x < alvo_x + alvo_tamanho and x + w > alvo_x and
            y < alvo_y + alvo_tamanho and y + h > alvo_y)

# Caminhos das imagens
caminho_imagem_balao = './img/balao.png'
caminho_imagem_estouro = './img/estouro.png'

def gen_frames():
    cap = cv2.VideoCapture(1)
    alvo_tamanho = 80

    # Intervalos de cor para detecção
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    score = 0
    imagem_alvo = cv2.imread(caminho_imagem_balao, cv2.IMREAD_UNCHANGED)
    imagem_estouro = cv2.imread(caminho_imagem_estouro, cv2.IMREAD_UNCHANGED)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return  # Sai se não conseguir capturar o frame
    frame_height, frame_width, _ = frame.shape
    alvo_posicao = gerar_posicao_aleatoria(frame_width, frame_height, alvo_tamanho, 150, (0, 0))

    tempo_ultima_mudanca = time.time()
    intervalo_mudanca = 7
    tempo_estouro = 0
    posicao_estouro = None
    fade_out_duracao = 0.5  # Duração do fade out em segundos

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break  # Libera a captura se falhar

            frame = cv2.flip(frame, 1)  # Inverte a imagem horizontalmente

            # Detecta as cores
            mask_azul = detectar_cor(frame, lower_blue, upper_blue)

            colisao_ocorreu = False

            # Processa contornos da máscara azul
            contours, _ = cv2.findContours(mask_azul, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if detectar_colisao(x, y, w, h, alvo_posicao[0], alvo_posicao[1], alvo_tamanho):
                        colisao_ocorreu = True
                    # Desenha o retângulo verde em vez de branco
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            if colisao_ocorreu:
                score += 1
                posicao_estouro = alvo_posicao
                alvo_posicao = gerar_posicao_aleatoria(frame_width, frame_height, alvo_tamanho, 150, alvo_posicao)
                tempo_ultima_mudanca = time.time()
                tempo_estouro = time.time()
                if score % 10 == 0 and intervalo_mudanca > 2:
                    intervalo_mudanca -= 1  # Diminui o tempo de intervalo a cada 10 pontos

            # Atualiza o tempo restante para o balão mudar
            tempo_restante = max(0, intervalo_mudanca - (time.time() - tempo_ultima_mudanca))
            if tempo_restante == 0:
                score -= 1
                alvo_posicao = gerar_posicao_aleatoria(frame_width, frame_height, alvo_tamanho, 150, alvo_posicao)
                tempo_ultima_mudanca = time.time()

            # Desenha o balão na imagem
            alvo = cv2.resize(imagem_alvo, (alvo_tamanho, alvo_tamanho))
            alpha_s = alvo[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(3):
                frame[alvo_posicao[1]:alvo_posicao[1] + alvo_tamanho, alvo_posicao[0]:alvo_posicao[0] + alvo_tamanho, c] = (
                    alpha_s * alvo[:, :, c] + alpha_l * frame[alvo_posicao[1]:alvo_posicao[1] + alvo_tamanho, alvo_posicao[0]:alvo_posicao[0] + alvo_tamanho, c])

            # Exibe a animação de estouro com fade out
            if posicao_estouro is not None:
                elapsed_time = time.time() - tempo_estouro
                if elapsed_time < fade_out_duracao:
                    fade_factor = 1 - (elapsed_time / fade_out_duracao)  # Gradualmente reduz a opacidade
                    estouro = cv2.resize(imagem_estouro, (alvo_tamanho, alvo_tamanho))
                    alpha_s_estouro = (estouro[:, :, 3] / 255.0) * fade_factor
                    alpha_l_estouro = 1.0 - alpha_s_estouro

                    for c in range(3):
                        frame[posicao_estouro[1]:posicao_estouro[1] + alvo_tamanho, posicao_estouro[0]:posicao_estouro[0] + alvo_tamanho, c] = (
                            alpha_s_estouro * estouro[:, :, c] + alpha_l_estouro * frame[posicao_estouro[1]:posicao_estouro[1] + alvo_tamanho, posicao_estouro[0]:posicao_estouro[0] + alvo_tamanho, c])
                else:
                    posicao_estouro = None  # Remove a explosão após o fade out

            # Exibe o score e o tempo restante
            cv2.putText(frame, f"Score: {score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Tempo: {int(tempo_restante)}s", (frame_width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Codifica o frame para a resposta do Flask
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Erro: {e}")
            break

    cap.release()  # Libera a captura no final

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
