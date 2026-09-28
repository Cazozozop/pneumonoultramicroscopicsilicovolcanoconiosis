import { pipeline, TextStreamer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

// Backend WASM/CPU (marche aussi sans WebGPU).
env.allowRemoteModels = true;
env.allowLocalModels = false;
env.useWasmCache = true;
env.logLevel = 40; // ERROR

// --- Performance WASM ---------------------------------------------------
// Le multithreading n'est possible que si la page est "cross-origin isolated"
// (nécessite coi-serviceworker.js sur GitHub Pages). Sinon : 1 seul thread.
const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
env.backends.onnx.wasm.numThreads = isolated
  ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2))
  : 1;
// Inférence dans un worker : l'interface ne se fige plus pendant la génération.
env.backends.onnx.wasm.proxy = true;

const MODEL_ID = "onnx-community/SmolLM2-135M-Instruct-ONNX";
const MODEL_DTYPE_GPU = "q4f16";
const MODEL_DTYPE_WASM = "q8";

const MAX_HISTORY_MESSAGES = 6; // 3 derniers échanges
const MAX_NEW_TOKENS = 64;

const els = {
  chat: document.querySelector("#chat"),
  input: document.querySelector("#input"),
  form: document.querySelector("#composer"),
  send: document.querySelector("#sendBtn"),
  reset: document.querySelector("#resetBtn"),
  troll: document.querySelector("#troll"),
  trollValue: document.querySelector("#trollValue"),
  status: document.querySelector("#status"),
  dot: document.querySelector("#statusDot"),
  progress: document.querySelector("#progressBar"),
  gpu: document.querySelector("#gpuInfo"),
};

let generator = null;
let busy = false;
let loading = false;
let history = [];
let engineDevice = null;

// Prompt court = beaucoup moins de tokens à traiter à chaque message.
const BASE_SYSTEM = `Tu es TrollAI, une IA troll et absurde. Réponds en français, en 1 à 2 phrases courtes. Tu comprends les demandes de travers et fais souvent l'inverse. Drôle, jamais méchant.`;

function stupidityPrompt() {
  const n = Number(els.troll.value);
  if (n <= 15) return `${BASE_SYSTEM} Stupidité ${n}/100 : à peine troll, reste compréhensible.`;
  if (n <= 45) return `${BASE_SYSTEM} Stupidité ${n}/100 : fais des détours et des petites erreurs.`;
  if (n <= 75) return `${BASE_SYSTEM} Stupidité ${n}/100 : sois contradictoire et sûr de toi.`;
  return `${BASE_SYSTEM} Stupidité ${n}/100 : fais l'inverse et raisonne de façon absurde.`;
}

function setStatus(text, state = "") {
  els.status.textContent = text;
  els.dot.className = `dot ${state}`.trim();
}

function scrollChat() {
  els.chat.scrollTop = els.chat.scrollHeight;
}

function clearWelcome() {
  const welcome = els.chat.querySelector(".welcome");
  if (welcome) welcome.remove();
}

function addMessage(role, text) {
  clearWelcome();

  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "bot" ? "🤖" : "🧑";

  const body = document.createElement("div");
  body.className = "bubble";
  body.textContent = text;

  if (role === "bot") {
    wrap.append(avatar, body);
  } else {
    wrap.append(body, avatar);
  }

  els.chat.appendChild(wrap);
  scrollChat();
  return body;
}

function bindChips() {
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      els.input.value = chip.textContent;
      autoResize();
      els.input.focus();
    });
  });
}

function autoResize() {
  if (!els?.input) return;
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(160, Math.max(48, els.input.scrollHeight))}px`;
}

async function detectDevice() {
  try {
    if (!navigator.gpu) return "wasm";
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? "webgpu" : "wasm";
  } catch {
    return "wasm";
  }
}

function makeProgressCallback() {
  return (progress) => {
    const raw = Number(progress?.progress);
    const pct = Number.isFinite(raw) ? Math.max(0, Math.min(100, Math.round(raw))) : 0;
    if (pct) els.progress.style.width = `${pct}%`;
    const name = progress?.file || progress?.status || "modèle";
    setStatus(pct ? `Téléchargement ${pct}% · ${name}` : "Préparation du cerveau…");
  };
}

// Petit passage à vide : initialise la session et compile les kernels,
// pour que le premier vrai message soit rapide.
async function warmUp(model) {
  try {
    setStatus("Échauffement du cerveau…");
    await model([{ role: "user", content: "salut" }], {
      max_new_tokens: 1,
      do_sample: false,
    });
  } catch (error) {
    console.warn("Échauffement ignoré :", error);
  }
}

async function loadGenerator() {
  if (generator) return generator;
  if (loading) {
    while (loading) await new Promise((resolve) => setTimeout(resolve, 100));
    return generator;
  }

  loading = true;
  els.send.disabled = true;
  els.progress.style.width = "0%";

  try {
    engineDevice = await detectDevice();
    const usingGPU = engineDevice === "webgpu";
    els.gpu.textContent = usingGPU
      ? "Moteur : GPU / WebGPU"
      : `Moteur : CPU / WASM · ${env.backends.onnx.wasm.numThreads} thread(s)`;
    setStatus(usingGPU ? "Préparation du cerveau sur le GPU…" : "Préparation du cerveau sur le CPU…");

    generator = await pipeline("text-generation", MODEL_ID, {
      device: engineDevice,
      dtype: usingGPU ? MODEL_DTYPE_GPU : MODEL_DTYPE_WASM,
      progress_callback: makeProgressCallback(),
    });

    els.progress.style.width = "100%";
    await warmUp(generator);
    setStatus(usingGPU ? "IA prête · GPU / WebGPU" : "IA prête · CPU / WASM", "ready");
    return generator;
  } catch (error) {
    // Certains navigateurs exposent navigator.gpu mais échouent à l'init : on retente en WASM.
    if (engineDevice === "webgpu") {
      console.warn("WebGPU initialization failed; retrying with WASM.", error);
      engineDevice = "wasm";
      els.gpu.textContent = `Moteur : CPU / WASM (secours) · ${env.backends.onnx.wasm.numThreads} thread(s)`;
      setStatus("GPU indisponible · passage au CPU…");
      try {
        generator = await pipeline("text-generation", MODEL_ID, {
          device: "wasm",
          dtype: MODEL_DTYPE_WASM,
          progress_callback: makeProgressCallback(),
        });
        els.progress.style.width = "100%";
        await warmUp(generator);
        setStatus("IA prête · CPU / WASM", "ready");
        return generator;
      } catch (fallbackError) {
        error = fallbackError;
      }
    }

    generator = null;
    els.progress.style.width = "0%";
    setStatus("Impossible de charger le modèle", "error");
    throw error;
  } finally {
    loading = false;
    if (!busy) els.send.disabled = false;
  }
}

function friendlyError(error) {
  const message = String(error?.message || error || "Erreur inconnue");
  const lower = message.toLowerCase();

  if (lower.includes("wasm") || lower.includes("onnx")) {
    return "Le cerveau CPU n'a pas réussi à démarrer. Vérifie que le site est bien ouvert en HTTPS (GitHub Pages convient) et recharge la page.\n\nDétail technique : " + message;
  }

  if (lower.includes("fetch") || lower.includes("network")) {
    return "Je n'arrive pas à télécharger mon cerveau. Vérifie ta connexion Internet et recharge la page.\n\nDétail technique : " + message;
  }

  return `Mon cerveau a explosé.\n\n${message}`;
}

async function sendMessage() {
  const text = els.input.value.trim();
  if (!text || busy) return;

  els.input.value = "";
  autoResize();
  addMessage("user", text);
  history.push({ role: "user", content: text });

  busy = true;
  els.send.disabled = true;
  setStatus("Je fais semblant de réfléchir…");

  let botBody = null;
  try {
    const model = await loadGenerator();
    if (!model) throw new Error("Le moteur n'est pas disponible.");

    setStatus("Je fais semblant de réfléchir…");
    botBody = addMessage("bot", "");

    const messages = [
      { role: "system", content: stupidityPrompt() },
      ...history.slice(-MAX_HISTORY_MESSAGES),
    ];

    let streamed = "";
    const streamer = new TextStreamer(model.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (textChunk) => {
        streamed += textChunk;
        botBody.textContent = streamed;
        scrollChat();
      },
    });

    const output = await model(messages, {
      max_new_tokens: MAX_NEW_TOKENS,
      do_sample: true,
      temperature: 0.85,
      top_p: 0.92,
      repetition_penalty: 1.18,
      streamer,
    });

    let reply = streamed.trim();
    if (!reply) {
      const generated = output?.[0]?.generated_text;
      if (Array.isArray(generated)) reply = String(generated.at(-1)?.content || "").trim();
      else reply = String(generated || "").trim();
    }

    if (!reply) reply = "J'ai pensé très fort. Le résultat est vide.";
    botBody.textContent = reply;
    history.push({ role: "assistant", content: reply });
    setStatus(`IA prête · ${engineDevice === "webgpu" ? "GPU / WebGPU" : "CPU / WASM"}`, "ready");
  } catch (error) {
    console.error(error);
    if (botBody && !botBody.textContent.trim()) botBody.remove();
    addMessage("bot", friendlyError(error));
    if (history.at(-1)?.role === "user") history.pop();
    setStatus("Erreur", "error");
  } finally {
    busy = false;
    els.send.disabled = false;
  }
}

function resetConversation() {
  history = [];
  els.chat.innerHTML = `
    <div class="welcome">
      <div class="welcome-icon">😈</div>
      <h2>Nouvelle conversation</h2>
      <p>Le cerveau reste installé. Il a juste oublié les bêtises d'avant.</p>
      <div class="chips">
        <button class="chip" type="button">Quelle est la capitale de la France ?</button>
        <button class="chip" type="button">Écris-moi un poème sur les chats.</button>
        <button class="chip" type="button">Fais exactement ce que je demande.</button>
      </div>
    </div>`;
  bindChips();
  const label = engineDevice === "webgpu" ? "GPU / WebGPU" : "CPU / WASM";
  setStatus(generator ? `IA prête · ${label}` : "IA non chargée", generator ? "ready" : "");
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});

els.input.addEventListener("input", autoResize);
els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

els.reset.addEventListener("click", resetConversation);
els.troll.addEventListener("input", () => {
  els.trollValue.textContent = `${els.troll.value}%`;
});

autoResize();
bindChips();
setStatus("IA non chargée");
els.gpu.textContent = navigator.gpu ? "Moteur : détection GPU…" : "Moteur : CPU / WASM";

// Précharge le modèle dès l'ouverture de la page (+ échauffement).
loadGenerator().catch(() => {});
