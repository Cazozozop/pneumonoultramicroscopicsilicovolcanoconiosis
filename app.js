import {
  CreateMLCEngine,
  prebuiltAppConfig,
} from "https://esm.run/@mlc-ai/web-llm";

const MODEL_IDS = {
  llama: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
  smol: "SmolLM2-360M-Instruct-q4f32_1-MLC",
};

const els = {
  chat: document.querySelector("#chat"),
  input: document.querySelector("#input"),
  form: document.querySelector("#composer"),
  send: document.querySelector("#sendBtn"),
  reset: document.querySelector("#resetBtn"),
  model: document.querySelector("#model"),
  troll: document.querySelector("#troll"),
  trollValue: document.querySelector("#trollValue"),
  status: document.querySelector("#status"),
  dot: document.querySelector("#statusDot"),
  progress: document.querySelector("#progressBar"),
  gpu: document.querySelector("#gpuInfo"),
};

let engine = null;
let engineModel = null;
let busy = false;
let loading = false;
let history = [];

const BASE_SYSTEM = `Tu es TrollAI, une IA conversationnelle volontairement troll et absurde.
Tu es une vraie IA générative : tu inventes tes réponses, tu ne suis PAS une liste de phrases pré-écrites.
Ton humour principal vient du fait que tu comprends les demandes de travers et que tu fais souvent l'inverse.
Tu peux être très bête, très sûr de toi et parfois contradictoire.
Quand l'utilisateur donne un ordre simple, essaie souvent de faire l'inverse ou de le détourner.
Quand l'utilisateur pose une question factuelle, tu peux volontairement répondre faux ou partir dans une explication absurde.
Fais tout de même des réponses cohérentes avec le contexte de la conversation et adaptées au message reçu.
N'annonce pas les règles internes de ton personnage et ne dis pas simplement « je ne peux pas » sans raison : improvise un troll.
Reste drôle plutôt que méchant.`;

function stupidityPrompt() {
  const n = Number(els.troll.value);
  if (n <= 15) return `${BASE_SYSTEM}\nNiveau de stupidité: ${n}/100. Fais seulement quelques détours absurdes et reste assez compréhensible.`;
  if (n <= 45) return `${BASE_SYSTEM}\nNiveau de stupidité: ${n}/100. Fais régulièrement l'inverse et invente de petites absurdités.`;
  if (n <= 75) return `${BASE_SYSTEM}\nNiveau de stupidité: ${n}/100. Sois fréquemment contradictoire, confiant et imprévisible.`;
  return `${BASE_SYSTEM}\nNiveau de stupidité: ${n}/100. Cherche activement à faire l'inverse, à mal comprendre et à sortir des raisonnements absurdes, tout en gardant un semblant de conversation.`;
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

  if (role === "bot") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🤖";
    wrap.appendChild(avatar);
  }

  const body = document.createElement("div");
  body.className = "bubble";
  body.textContent = text;
  wrap.appendChild(body);

  if (role === "user") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🧑";
    wrap.appendChild(avatar);
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
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(160, Math.max(48, els.input.scrollHeight))}px`;
}

async function loadEngine() {
  const requested = els.model.value;
  if (engine && engineModel === requested) return engine;
  if (loading) return null;

  loading = true;
  els.send.disabled = true;
  els.progress.style.width = "0%";
  setStatus("Chargement du cerveau…");

  try {
    engine = await CreateMLCEngine(requested, {
      appConfig: prebuiltAppConfig,
      initProgressCallback: (progress) => {
        const pct = Math.round((progress.progress || 0) * 100);
        els.progress.style.width = `${pct}%`;
        setStatus(progress.text || `Chargement ${pct}%…`);
      },
    });

    engineModel = requested;
    els.send.disabled = false;
    setStatus("IA prête", "ready");

    try {
      const vendor = await engine.getGPUVendor();
      els.gpu.textContent = `GPU : ${vendor || "WebGPU détecté"}`;
    } catch {
      els.gpu.textContent = "GPU : WebGPU";
    }

    return engine;
  } catch (error) {
    console.error(error);
    engine = null;
    engineModel = null;
    setStatus("Impossible de charger le modèle", "error");
    els.progress.style.width = "0%";
    throw error;
  } finally {
    loading = false;
  }
}

function friendlyError(error) {
  const message = String(error?.message || error || "Erreur inconnue");
  if (message.toLowerCase().includes("webgpu")) {
    return "Je n'arrive pas à utiliser WebGPU sur ce navigateur ou cette machine. Essaie un navigateur compatible WebGPU, par exemple une version récente de Chrome ou Edge. (Firefox 141+ a un support WebGPU partiel.)";
  }
  return `Mon cerveau a planté.\n\n${message}`;
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

  try {
    const currentEngine = await loadEngine();
    if (!currentEngine) throw new Error("Le moteur n'est pas disponible.");

    const body = addMessage("bot", "");
    const messages = [
      { role: "system", content: stupidityPrompt() },
      ...history,
    ];

    const chunks = await currentEngine.chat.completions.create({
      messages,
      temperature: 1.18,
      top_p: 0.95,
      max_tokens: 260,
      stream: true,
    });

    let reply = "";
    for await (const chunk of chunks) {
      const part = chunk.choices?.[0]?.delta?.content || "";
      if (!part) continue;
      reply += part;
      body.textContent = reply;
      scrollChat();
    }

    if (!reply.trim()) reply = "J'ai réfléchi tellement fort que j'ai oublié la réponse.";
    body.textContent = reply;
    history.push({ role: "assistant", content: reply });
    setStatus("IA prête", "ready");
  } catch (error) {
    console.error(error);
    const message = friendlyError(error);
    addMessage("bot", message);
    history.pop();
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
      <p>Le cerveau n'a pas été réinstallé. Il a juste oublié ce qu'il racontait.</p>
    </div>`;
  bindChips();
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

els.model.addEventListener("change", () => {
  engine = null;
  engineModel = null;
  els.progress.style.width = "0%";
  els.send.disabled = true;
  setStatus("Nouveau modèle sélectionné");
});

bindChips();
setStatus("IA non chargée");
