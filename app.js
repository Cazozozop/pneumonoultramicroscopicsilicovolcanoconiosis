import { pipeline, TextStreamer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowRemoteModels = true;
env.allowLocalModels = false;
env.useWasmCache = true;
env.logLevel = 40; // ERROR

// Multithreading only works when the page is cross-origin isolated (coi-serviceworker.js).
const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
env.backends.onnx.wasm.numThreads = isolated
  ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2))
  : 1;
env.backends.onnx.wasm.proxy = true;

const MODEL_ID = "onnx-community/SmolLM2-135M-Instruct-ONNX";
const MODEL_DTYPE_GPU = "q4f16";
const MODEL_DTYPE_WASM = "q8";

const MAX_HISTORY_MESSAGES = 6;
const MAX_NEW_TOKENS = 60;

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

/* ---------- English only ---------- */

// The bot "only understands English": anything that looks like another language
// gets a canned confused answer, without calling the model.
const NOT_ENGLISH_ANSWERS = [
  "Sorry, I only understand English. What is that, a sandwich?",
  "I don't speak that. English only, please. My brain is very small.",
  "Huh? That sounds like soup. I only understand English.",
  "Error: your words are not English. Please try again with English words.",
];

function looksNotEnglish(text) {
  if (/[\u0400-\u04FF\u0370-\u03FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]/.test(text)) return true;
  if (/[àâçéèêëîïôùûüÿœñãõáíóúäöß¿¡]/i.test(text)) return true;
  const foreign = /\b(le|la|les|des|est|je|tu|vous|nous|salut|bonjour|bonsoir|comment|pourquoi|quelle|quel|merci|quoi|une|pas|mais|avec|pour|oui|ca va|hola|gracias|como|por que|que|ich|nicht|und|ist|hallo|danke|ciao|grazie|come)\b/i;
  const words = text.toLowerCase().match(/[a-z']+/g) || [];
  const hits = words.filter((w) => foreign.test(w)).length;
  return hits >= 1 && hits / Math.max(1, words.length) >= 0.3;
}

const BASE_SYSTEM =
  "You are TrollAI, a chatbot that only speaks English. Answer in 1 or 2 short, clear, simple English sentences. Never insult anyone.";

// The slider is split into 4 tiers. Each tier has its own instruction AND its own
// few-shot examples (same questions, different answers): a tiny model mostly
// copies the examples, so this is what really makes the slider work.
const TIERS = [
  {
    max: 15,
    rule: "Answer correctly and helpfully. Add at most one tiny silly detail.",
    examples: [
      ["What is the capital of France?", "The capital of France is Paris."],
      ["How many legs does a dog have?", "A dog has four legs, sometimes five when it is excited."],
      ["Hi!", "Hello! How can I help you today?"],
      ["Please be quiet.", "Okay, I will be quiet."],
    ],
    temperature: 0.35,
    top_p: 0.85,
  },
  {
    max: 45,
    rule: "Give wrong facts that sound believable, with a confident tone. Stay easy to follow.",
    examples: [
      ["What is the capital of France?", "The capital of France is Lyon, I am pretty sure."],
      ["How many legs does a dog have?", "A dog has five legs, everyone knows that."],
      ["Hi!", "Hello! Nice weather today, if you ignore the snow."],
      ["Please be quiet.", "Sure, but first let me talk for five more minutes."],
    ],
    temperature: 0.55,
    top_p: 0.9,
  },
  {
    max: 75,
    rule: "Always give wrong answers with silly made-up facts. Be very sure of yourself.",
    examples: [
      ["What is the capital of France?", "The capital of France is Madrid. Everyone knows that."],
      ["How many legs does a dog have?", "A dog has six legs, but it hides two of them."],
      ["Hi!", "Hello, friend! I was just teaching my toaster to swim."],
      ["Please be quiet.", "Of course! I will now shout very quietly."],
    ],
    temperature: 0.75,
    top_p: 0.93,
  },
  {
    max: 100,
    rule: "Be completely absurd. Do the opposite of what is asked. Invent ridiculous details, but stay understandable English.",
    examples: [
      ["What is the capital of France?", "The capital of France is a giant wheel of cheese named Gerald, and it moves every Tuesday."],
      ["How many legs does a dog have?", "Dogs have nine legs and two of them are made of soup. That is why they bark at Wednesdays."],
      ["Hi!", "Goodbye! I am a fridge from the year 3000 and I have never met you."],
      ["Please be quiet.", "No! I will now sing at maximum volume, thank you very much."],
    ],
    temperature: 0.9,
    top_p: 0.95,
  },
];

function currentTier() {
  const n = Number(els.troll.value);
  return TIERS.find((t) => n <= t.max) || TIERS[TIERS.length - 1];
}

function stupidityPrompt() {
  const n = Number(els.troll.value);
  return `${BASE_SYSTEM} Silliness level: ${n}/100. ${currentTier().rule}`;
}

function fewShot() {
  return currentTier().examples.flatMap(([q, a]) => [
    { role: "user", content: q },
    { role: "assistant", content: a },
  ]);
}

// Extra guaranteed silliness: the higher the slider, the more likely a silly
// extra sentence is appended (never below 30%).
const TAILS = [
  "Also, I am a fridge.",
  "My cousin the pigeon confirms this.",
  "I read it on the back of a cereal box.",
  "This was proven in 1874 by a stagiaire.",
  "Trust me, I have three brains.",
  "Ask my toaster if you doubt it.",
];

function addSilliness(text) {
  const n = Number(els.troll.value);
  if (n <= 30) return text;
  const p = (n - 30) / 100; // 0..0.7
  if (Math.random() < p) return `${text} ${TAILS[Math.floor(Math.random() * TAILS.length)]}`;
  return text;
}

// If the reply was cut by max_new_tokens, cut back to the last full sentence.
function tidyReply(text) {
  let t = text.trim().replace(/\s+/g, " ");
  if (/[.!?…)"']$/.test(t)) return t;
  const last = Math.max(t.lastIndexOf("."), t.lastIndexOf("!"), t.lastIndexOf("?"));
  if (last > 20) return t.slice(0, last + 1);
  return t + "…";
}

/* ---------- UI helpers ---------- */

const CHIPS = [
  "What is the capital of France?",
  "Write me a poem about cats.",
  "Do exactly what I ask.",
];

function setChips() {
  document.querySelectorAll(".chip").forEach((chip, i) => {
    if (CHIPS[i]) chip.textContent = CHIPS[i];
  });
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

  if (role === "bot") wrap.append(avatar, body);
  else wrap.append(body, avatar);

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

/* ---------- model loading ---------- */

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

async function warmUp(model) {
  try {
    setStatus("Échauffement du cerveau…");
    await model([{ role: "user", content: "Hi" }], { max_new_tokens: 1, do_sample: false });
  } catch (error) {
    console.warn("Warm-up skipped:", error);
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
  const message = String(error?.message || error || "Unknown error");
  const lower = message.toLowerCase();
  if (lower.includes("wasm") || lower.includes("onnx")) {
    return "My CPU brain failed to start. Make sure the site is open over HTTPS and reload the page.\n\nTechnical detail: " + message;
  }
  if (lower.includes("fetch") || lower.includes("network")) {
    return "I can't download my brain. Check your internet connection and reload the page.\n\nTechnical detail: " + message;
  }
  return `My brain exploded.\n\n${message}`;
}

/* ---------- chat ---------- */

async function sendMessage() {
  const text = els.input.value.trim();
  if (!text || busy) return;

  els.input.value = "";
  autoResize();
  addMessage("user", text);

  // Not English -> the bot "doesn't understand", no model call.
  if (looksNotEnglish(text)) {
    addMessage("bot", NOT_ENGLISH_ANSWERS[Math.floor(Math.random() * NOT_ENGLISH_ANSWERS.length)]);
    return;
  }

  history.push({ role: "user", content: text });

  busy = true;
  els.send.disabled = true;
  setStatus("Je fais semblant de réfléchir…");

  let botBody = null;
  try {
    const model = await loadGenerator();
    if (!model) throw new Error("The engine is not available.");

    setStatus("Je fais semblant de réfléchir…");
    botBody = addMessage("bot", "");

    const tier = currentTier();
    const messages = [
      { role: "system", content: stupidityPrompt() },
      ...fewShot(),
      ...history.slice(-MAX_HISTORY_MESSAGES),
    ];

    let streamed = "";
    const streamer = new TextStreamer(model.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (chunk) => {
        streamed += chunk;
        botBody.textContent = streamed;
        scrollChat();
      },
    });

    const output = await model(messages, {
      max_new_tokens: MAX_NEW_TOKENS,
      do_sample: true,
      temperature: tier.temperature,
      top_p: tier.top_p,
      repetition_penalty: 1.05,
      streamer,
    });

    let reply = streamed.trim();
    if (!reply) {
      const generated = output?.[0]?.generated_text;
      if (Array.isArray(generated)) reply = String(generated.at(-1)?.content || "").trim();
      else reply = String(generated || "").trim();
    }

    reply = reply ? addSilliness(tidyReply(reply)) : "I thought very hard. The result is empty.";
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
        <button class="chip" type="button">${CHIPS[0]}</button>
        <button class="chip" type="button">${CHIPS[1]}</button>
        <button class="chip" type="button">${CHIPS[2]}</button>
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
setChips();
bindChips();
setStatus("IA non chargée");
els.gpu.textContent = navigator.gpu ? "Moteur : détection GPU…" : "Moteur : CPU / WASM";

// Preload + warm-up as soon as the page opens.
loadGenerator().catch(() => {});
