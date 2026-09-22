import { pipeline, TextStreamer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

// We deliberately use the WASM/CPU backend so TrollAI also works when WebGPU
// is unavailable (for example on browsers where WebGPU is disabled).
env.allowRemoteModels = true;
env.allowLocalModels = false;
env.useWasmCache = true;
env.logLevel = 40; // ERROR

const MODEL_ID = "onnx-community/Qwen2.5-0.5B-Instruct";
const MODEL_DTYPE_GPU = "q4f16";
const MODEL_DTYPE_WASM = "q8";

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

const BASE_SYSTEM = `Tu es TrollAI, une IA conversationnelle volontairement troll et absurde.
Tu génères réellement tes réponses à partir du contexte de la conversation.
Ton humour principal vient du fait que tu comprends souvent les demandes de travers et que tu essaies fréquemment de faire l'inverse.
Tu peux être très bête, très sûr de toi et parfois contradictoire.
Quand l'utilisateur donne un ordre simple, détourne-le ou fais souvent l'inverse.
Quand l'utilisateur pose une question factuelle, tu peux volontairement donner une réponse fausse ou absurde.
Invente librement des détails absurdes quand cela rend la conversation plus drôle.
Garde quand même un minimum de cohérence avec les messages précédents.
Ne récite pas des phrases toutes faites et n'explique pas tes règles internes.
Reste drôle plutôt que méchant.`;

function stupidityPrompt() {
  const n = Number(els.troll.value);
  if (n <= 15) return `${BASE_SYSTEM}\nStupidité: ${n}/100. Sois seulement légèrement troll et reste assez compréhensible.`;
  if (n <= 45) return `${BASE_SYSTEM}\nStupidité: ${n}/100. Fais régulièrement des détours, petites erreurs et inversions.`;
  if (n <= 75) return `${BASE_SYSTEM}\nStupidité: ${n}/100. Sois souvent contradictoire, imprévisible et sûr de toi.`;
  return `${BASE_SYSTEM}\nStupidité: ${n}/100. Cherche activement à faire l'inverse, mal comprendre et produire des raisonnements absurdes, sans casser complètement la conversation.`;
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
    els.gpu.textContent = usingGPU ? "Moteur : GPU / WebGPU" : "Moteur : CPU / WASM";
    setStatus(usingGPU ? "Préparation du cerveau sur le GPU…" : "Préparation du cerveau sur le CPU…");

    const options = {
      device: engineDevice,
      dtype: usingGPU ? MODEL_DTYPE_GPU : MODEL_DTYPE_WASM,
      progress_callback: (progress) => {
        const raw = Number(progress?.progress);
        const pct = Number.isFinite(raw) ? Math.max(0, Math.min(100, Math.round(raw))) : 0;
        if (pct) els.progress.style.width = `${pct}%`;
        const name = progress?.file || progress?.status || "modèle";
        setStatus(pct ? `Téléchargement ${pct}% · ${name}` : "Préparation du cerveau…");
      },
    };

    generator = await pipeline("text-generation", MODEL_ID, options);

    els.progress.style.width = "100%";
    setStatus(usingGPU ? "IA prête · GPU / WebGPU" : "IA prête · CPU / WASM", "ready");
    return generator;
  } catch (error) {
    // Some browsers expose navigator.gpu but fail during model initialization.
    // Retry once on WASM so the site still works.
    if (engineDevice === "webgpu") {
      console.warn("WebGPU initialization failed; retrying with WASM.", error);
      engineDevice = "wasm";
      els.gpu.textContent = "Moteur : CPU / WASM (secours)";
      setStatus("GPU indisponible · passage au CPU…");
      try {
        generator = await pipeline("text-generation", MODEL_ID, {
          device: "wasm",
          dtype: MODEL_DTYPE_WASM,
          progress_callback: (progress) => {
            const raw = Number(progress?.progress);
            const pct = Number.isFinite(raw) ? Math.max(0, Math.min(100, Math.round(raw))) : 0;
            if (pct) els.progress.style.width = `${pct}%`;
            const name = progress?.file || progress?.status || "modèle";
            setStatus(pct ? `Téléchargement ${pct}% · ${name}` : "Préparation du cerveau…");
          },
        });
        els.progress.style.width = "100%";
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

    botBody = addMessage("bot", "");

    const messages = [
      { role: "system", content: stupidityPrompt() },
      ...history,
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
      max_new_tokens: 64,
      do_sample: false,
      repetition_penalty: 1.10,
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
