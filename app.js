import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

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
  if (/[\u0400-\u04FF\u0370-\u03FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]/.test(text)) {
    return true;
  }

  if (/[àâçéèêëîïôùûüÿœñãõáíóúäöß¿¡]/i.test(text)) {
    return true;
  }

  const foreign =
    /\b(le|la|les|des|est|je|tu|vous|nous|salut|bonjour|bonsoir|comment|pourquoi|quelle|quel|merci|quoi|une|pas|mais|avec|pour|oui|ca va|hola|gracias|como|por que|que|ich|nicht|und|ist|hallo|danke|ciao|grazie|come)\b/i;

  const words = text.toLowerCase().match(/[a-z']+/g) || [];
  const hits = words.filter((w) => foreign.test(w)).length;

  return hits >= 1 && hits / Math.max(1, words.length) >= 0.3;
}

const BASE_SYSTEM =
  "You are TrollAI, a chatbot that only speaks English. Answer in 1 or 2 short, clear, simple English sentences. Never insult anyone. Stay relevant to the user's question.";

// The slider controls how unreliable/silly the bot becomes.
// IMPORTANT: the bot should still sound intelligent and coherent at high values.
const TIERS = [
  {
    rule:
      "Answer correctly and helpfully. Stay natural and do not add unnecessary jokes.",
    examples: [
      ["What is the capital of France?", "The capital of France is Paris."],
      ["How many legs does a dog have?", "A dog has four legs."],
      ["Hi!", "Hello! How can I help you today?"],
      ["Please be quiet.", "Okay, I will be quiet."],
    ],
    max: 20,
    temperature: 0.35,
    top_p: 0.85,
  },

  {
    rule:
      "Answer normally and intelligently. You may make one small believable factual mistake, but most of the answer should remain correct.",
    examples: [
      ["What is the capital of France?", "The capital of France is Paris."],
      ["How many legs does a dog have?", "A dog has four legs."],
      ["Hi!", "Hello! How can I help you today?"],
      ["Please be quiet.", "Sure, I will be quiet."],
    ],
    max: 45,
    temperature: 0.45,
    top_p: 0.88,
  },

  {
    rule:
      "Give a believable answer that mostly makes sense. Sometimes include one incorrect factual detail. Do not become random or nonsensical.",
    examples: [
      ["What is the capital of France?", "The capital of France is Paris."],
      ["How many legs does a dog have?", "A dog has four legs."],
      ["What is water made of?", "Water is made of hydrogen and oxygen."],
      ["Hi!", "Hello! How can I help you?"],
    ],
    max: 70,
    temperature: 0.55,
    top_p: 0.9,
  },

  {
    rule:
      "Give a confident, coherent answer. It is okay to give one or two false facts, but the answer must still be relevant and intelligent. Never become completely random.",
    examples: [
      ["What is the capital of France?", "The capital of France is Paris."],
      ["How many legs does a dog have?", "A dog has four legs."],
      ["What is water made of?", "Water is made of hydrogen and oxygen."],
      ["Hi!", "Hello! What can I help you with?"],
    ],
    max: 100,
    temperature: 0.65,
    top_p: 0.92,
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

/* ---------- making answers slightly FALSE ---------- */

// The model itself stays mostly normal.
// At higher slider values, we selectively corrupt small factual details.

const rnd = (n) => Math.floor(Math.random() * n);
const pick = (arr) => arr[rnd(arr.length)];

function lieChance() {
  const n = Number(els.troll.value);

  // Very low chance at 0%.
  // Gradually increases to around 70% at 100%.
  return 0.02 + (0.68 * n) / 100;
}

const SWAPS = [
  ["Paris", "Madrid", "Rome", "Berlin", "London", "Tokyo", "Lisbon", "Vienna", "Cairo", "Moscow", "Lyon"],
  ["France", "Spain", "Italy", "Germany", "Japan", "Brazil", "Canada", "Egypt", "Peru", "Sweden"],
  ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white"],
  ["dog", "cat", "horse", "cow", "pig", "duck", "rabbit", "sheep", "goat"],
  ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Neptune"],
  ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
  ["north", "south", "east", "west"],
  ["hot", "cold"],
  ["big", "small"],
  ["sun", "moon"],
  ["day", "night"],
  ["fast", "slow"],
  ["true", "false"],
  ["always", "never"],
  ["summer", "winter"],
  ["morning", "evening"],
  ["water", "fire"],
  ["more", "less"],
  ["before", "after"],
];

const NUMW = [
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
];

function matchCase(orig, rep) {
  if (orig.length > 1 && orig === orig.toUpperCase()) {
    return rep.toUpperCase();
  }

  if (
    orig[0] === orig[0].toUpperCase() &&
    orig[0] !== orig[0].toLowerCase()
  ) {
    return rep[0].toUpperCase() + rep.slice(1);
  }

  return rep.toLowerCase() === rep && rep.length ? rep : rep;
}

function fmtNum(n) {
  return Number.isInteger(n)
    ? String(n)
    : String(Math.round(n * 100) / 100);
}

function wrongNumber(correct) {
  const n = Number(els.troll.value);

  // Keep mistakes relatively believable at lower levels.
  // High levels can produce larger mistakes.
  const spread = n > 75 ? 40 : n > 45 ? 12 : 5;

  let w = correct;
  let guard = 0;

  while (w === correct && guard++ < 20) {
    w = correct + (rnd(spread * 2 + 1) - spread);

    if (Math.random() < 0.25) {
      w = Math.round(correct * pick([0.5, 2, 10, -1]));
    }
  }

  return w === correct ? correct + 1 : w;
}

function falsify(reply, userText) {
  // Sometimes don't lie at all.
  if (Math.random() > lieChance()) {
    return reply;
  }

  const n = Number(els.troll.value);

  // Keep the number of changes small.
  const maxEdits =
    n <= 45
      ? 1
      : n <= 75
        ? 2
        : 2;

  const q = userText.toLowerCase();
  const edits = [];

  // Numbers.
  for (const m of reply.matchAll(/\b\d+(?:[.,]\d+)?\b/g)) {
    // Don't change numbers directly mentioned by the user.
    if (q.includes(m[0])) continue;

    edits.push({
      i: m.index,
      len: m[0].length,
      rep: fmtNum(
        wrongNumber(parseFloat(m[0].replace(",", ".")))
      ),
    });
  }

  // Written numbers.
  for (const m of reply.matchAll(
    /\b(two|three|four|five|six|seven|eight|nine|ten)\b/gi
  )) {
    if (q.includes(m[0].toLowerCase())) continue;

    edits.push({
      i: m.index,
      len: m[0].length,
      rep: matchCase(
        m[0],
        pick(NUMW.filter((w) => w !== m[0].toLowerCase()))
      ),
    });
  }

  // Simple semantic swaps.
  for (const group of SWAPS) {
    for (const w of group) {
      if (q.includes(w.toLowerCase())) continue;

      for (const m of reply.matchAll(new RegExp(`\\b${w}\\b`, "gi"))) {
        const others = group.filter(
          (x) =>
            x.toLowerCase() !== w.toLowerCase() &&
            !q.includes(x.toLowerCase())
        );

        if (others.length) {
          edits.push({
            i: m.index,
            len: m[0].length,
            rep: matchCase(m[0], pick(others)),
          });
        }
      }
    }
  }

  // Yes/no inversion.
  const yn = reply.match(/^(yes|no)\b/i);

  if (yn) {
    edits.push({
      i: 0,
      len: yn[0].length,
      rep: matchCase(
        yn[0],
        yn[1].toLowerCase() === "yes" ? "no" : "yes"
      ),
    });
  }

  // Negation.
  const neg = reply.match(/\b(is|are|was|were) not\b/i);

  if (neg) {
    edits.push({
      i: neg.index,
      len: neg[0].length,
      rep: neg[1],
    });
  } else {
    const pos = reply.match(/\b(is|are|was|were)\b/i);

    if (pos) {
      edits.push({
        i: pos.index + pos[0].length,
        len: 0,
        rep: " not",
      });
    }
  }

  // Choose a few non-overlapping edits.
  const chosen = [];

  for (const e of edits.sort(() => Math.random() - 0.5)) {
    if (chosen.length >= maxEdits) break;

    if (
      chosen.every(
        (c) =>
          e.i + e.len <= c.i ||
          e.i >= c.i + c.len
      )
    ) {
      chosen.push(e);
    }
  }

  // IMPORTANT:
  // If there is nothing sensible to falsify, keep the original answer.
  // Never invent a completely unrelated sentence.
  if (!chosen.length) {
    return reply;
  }

  let out = reply;

  for (const e of chosen.sort((a, b) => b.i - a.i)) {
    out =
      out.slice(0, e.i) +
      e.rep +
      out.slice(e.i + e.len);
  }

  return out;
}

/* ---------- silly endings ---------- */

// This is deliberately separate from factual lying.
// It gives the slider a fun "troll" personality without destroying the answer.

const SILLY_ENDINGS = [
  " I think.",
  " Probably.",
  " At least, that's what I remember.",
  " For some reason.",
  " Don't ask me why.",
  " My brain says so.",
  " I am fairly sure about that.",
  " Anyway.",
];

function maybeAddSillyEnding(reply) {
  const n = Number(els.troll.value);

  // Almost never at low slider values.
  if (n < 25) {
    return reply;
  }

  // Slowly increase the chance.
  const chance = (n - 20) / 160;

  if (Math.random() > chance) {
    return reply;
  }

  let out = reply.trim();

  // Avoid weird double punctuation.
  out = out.replace(/[\s]+$/, "");

  // Don't add an ending after very short empty answers.
  if (out.length < 4) {
    return out;
  }

  // If the model didn't end with punctuation, add a period.
  if (!/[.!?…]$/.test(out)) {
    out += ".";
  }

  // Don't stack multiple silly endings.
  for (const ending of SILLY_ENDINGS) {
    if (out.toLowerCase().endsWith(ending.toLowerCase())) {
      return out;
    }
  }

  return out + pick(SILLY_ENDINGS);
}

/* ---------- direct factual rules ---------- */

const CAPITALS = {
  france: "Paris",
  spain: "Madrid",
  italy: "Rome",
  germany: "Berlin",
  japan: "Tokyo",
  china: "Beijing",
  england: "London",
  "united kingdom": "London",
  uk: "London",
  "united states": "Washington",
  usa: "Washington",
  canada: "Ottawa",
  brazil: "Brasilia",
  switzerland: "Bern",
  belgium: "Brussels",
  portugal: "Lisbon",
  russia: "Moscow",
  india: "New Delhi",
  australia: "Canberra",
  morocco: "Rabat",
  egypt: "Cairo",
  mexico: "Mexico City",
  greece: "Athens",
};

function ruleAnswer(text) {
  const t = text.toLowerCase().trim();
  const lies = Math.random() < lieChance();

  // Basic arithmetic.
  let m = t
    .replace(/(\d),(\d)/g, "$1.$2")
    .match(
      /(-?\d+(?:\.\d+)?)\s*(\+|plus|-|minus|\*|x|times|\/|divided by)\s*(-?\d+(?:\.\d+)?)/
    );

  if (m) {
    const a = parseFloat(m[1]);
    const b = parseFloat(m[3]);
    const op = m[2];

    let c;

    if (op === "+" || op === "plus") {
      c = a + b;
    } else if (op === "-" || op === "minus") {
      c = a - b;
    } else if (op === "*" || op === "x" || op === "times") {
      c = a * b;
    } else {
      c = b === 0 ? NaN : a / b;
    }

    if (!Number.isFinite(c)) {
      return null;
    }

    c = Math.round(c * 100) / 100;

    const ans = fmtNum(lies ? wrongNumber(c) : c);

    return pick([
      `It is ${ans}.`,
      `The answer is ${ans}.`,
      `That is easy, it is ${ans}.`,
    ]);
  }

  // Capital of X.
  m = t
    .match(/capital (?:city )?of (?:the )?([a-z ]+?)\s*[?.!]*$/);

  if (m && CAPITALS[m[1].trim()]) {
    const real = CAPITALS[m[1].trim()];
    const country = m[1]
      .trim()
      .replace(/\b\w/g, (c) => c.toUpperCase());

    const others = [
      ...new Set(Object.values(CAPITALS)),
    ].filter((c) => c !== real);

    return `The capital of ${country} is ${
      lies ? pick(others) : real
    }.`;
  }

  return null;
}

/* ---------- response cleanup ---------- */

// If the reply was cut by max_new_tokens, cut back to the last full sentence.
function tidyReply(text) {
  let t = text.trim().replace(/\s+/g, " ");

  if (/[.!?…)"']$/.test(t)) {
    return t;
  }

  const last = Math.max(
    t.lastIndexOf("."),
    t.lastIndexOf("!"),
    t.lastIndexOf("?")
  );

  if (last > 20) {
    return t.slice(0, last + 1);
  }

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
    if (CHIPS[i]) {
      chip.textContent = CHIPS[i];
    }
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

  if (welcome) {
    welcome.remove();
  }
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
  if (!els?.input) {
    return;
  }

  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(
    160,
    Math.max(48, els.input.scrollHeight)
  )}px`;
}

/* ---------- model loading ---------- */

async function detectDevice() {
  try {
    if (!navigator.gpu) {
      return "wasm";
    }

    const adapter = await navigator.gpu.requestAdapter();

    return adapter ? "webgpu" : "wasm";
  } catch {
    return "wasm";
  }
}

function makeProgressCallback() {
  return (progress) => {
    const raw = Number(progress?.progress);

    const pct = Number.isFinite(raw)
      ? Math.max(0, Math.min(100, Math.round(raw)))
      : 0;

    if (pct) {
      els.progress.style.width = `${pct}%`;
    }

    const name =
      progress?.file ||
      progress?.status ||
      "modèle";

    setStatus(
      pct
        ? `Téléchargement ${pct}% · ${name}`
        : "Préparation du cerveau…"
    );
  };
}

async function warmUp(model) {
  try {
    setStatus("Échauffement du cerveau…");

    await model(
      [
        {
          role: "user",
          content: "Hi",
        },
      ],
      {
        max_new_tokens: 1,
        do_sample: false,
      }
    );
  } catch (error) {
    console.warn("Warm-up skipped:", error);
  }
}

async function loadGenerator() {
  if (generator) {
    return generator;
  }

  if (loading) {
    while (loading) {
      await new Promise((resolve) =>
        setTimeout(resolve, 100)
      );
    }

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

    setStatus(
      usingGPU
        ? "Préparation du cerveau sur le GPU…"
        : "Préparation du cerveau sur le CPU…"
    );

    generator = await pipeline(
      "text-generation",
      MODEL_ID,
      {
        device: engineDevice,
        dtype: usingGPU
          ? MODEL_DTYPE_GPU
          : MODEL_DTYPE_WASM,
        progress_callback:
          makeProgressCallback(),
      }
    );

    els.progress.style.width = "100%";

    await warmUp(generator);

    setStatus(
      usingGPU
        ? "IA prête · GPU / WebGPU"
        : "IA prête · CPU / WASM",
      "ready"
    );

    return generator;
  } catch (error) {
    if (engineDevice === "webgpu") {
      console.warn(
        "WebGPU initialization failed; retrying with WASM.",
        error
      );

      engineDevice = "wasm";

      els.gpu.textContent =
        `Moteur : CPU / WASM (secours) · ${env.backends.onnx.wasm.numThreads} thread(s)`;

      setStatus(
        "GPU indisponible · passage au CPU…"
      );

      try {
        generator = await pipeline(
          "text-generation",
          MODEL_ID,
          {
            device: "wasm",
            dtype: MODEL_DTYPE_WASM,
            progress_callback:
              makeProgressCallback(),
          }
        );

        els.progress.style.width = "100%";

        await warmUp(generator);

        setStatus(
          "IA prête · CPU / WASM",
          "ready"
        );

        return generator;
      } catch (fallbackError) {
        error = fallbackError;
      }
    }

    generator = null;
    els.progress.style.width = "0%";
    setStatus(
      "Impossible de charger le modèle",
      "error"
    );

    throw error;
  } finally {
    loading = false;

    if (!busy) {
      els.send.disabled = false;
    }
  }
}

function friendlyError(error) {
  const message = String(
    error?.message ||
    error ||
    "Unknown error"
  );

  const lower = message.toLowerCase();

  if (
    lower.includes("wasm") ||
    lower.includes("onnx")
  ) {
    return (
      "My CPU brain failed to start. Make sure the site is open over HTTPS and reload the page.\n\nTechnical detail: " +
      message
    );
  }

  if (
    lower.includes("fetch") ||
    lower.includes("network")
  ) {
    return (
      "I can't download my brain. Check your internet connection and reload the page.\n\nTechnical detail: " +
      message
    );
  }

  return `My brain exploded.\n\n${message}`;
}

/* ---------- chat ---------- */

async function typeInto(el, text) {
  let i = 0;

  while (i < text.length) {
    i = Math.min(text.length, i + 2);

    el.textContent = text.slice(0, i);

    scrollChat();

    await new Promise((r) =>
      setTimeout(r, 12)
    );
  }
}

async function sendMessage() {
  const text = els.input.value.trim();

  if (!text || busy) {
    return;
  }

  els.input.value = "";
  autoResize();

  addMessage("user", text);

  // Not English -> the bot "doesn't understand", no model call.
  if (looksNotEnglish(text)) {
    addMessage(
      "bot",
      NOT_ENGLISH_ANSWERS[
        Math.floor(
          Math.random() *
            NOT_ENGLISH_ANSWERS.length
        )
      ]
    );

    return;
  }

  history.push({
    role: "user",
    content: text,
  });

  busy = true;
  els.send.disabled = true;

  setStatus("Je fais semblant de réfléchir…");

  let botBody = null;

  try {
    botBody = addMessage("bot", "…");

    let reply = ruleAnswer(text);

    if (!reply) {
      const model = await loadGenerator();

      if (!model) {
        throw new Error(
          "The engine is not available."
        );
      }

      setStatus(
        "Je fais semblant de réfléchir…"
      );

      const tier = currentTier();

      const messages = [
        {
          role: "system",
          content: stupidityPrompt(),
        },

        ...fewShot(),

        ...history.slice(
          -MAX_HISTORY_MESSAGES
        ),
      ];

      const output = await model(messages, {
        max_new_tokens: MAX_NEW_TOKENS,
        do_sample: true,
        temperature: tier.temperature,
        top_p: tier.top_p,
        repetition_penalty: 1.05,
      });

      const generated =
        output?.[0]?.generated_text;

      const raw = Array.isArray(generated)
        ? String(
            generated.at(-1)?.content || ""
          ).trim()
        : String(generated || "").trim();

      if (raw) {
        // First clean the model response.
        reply = tidyReply(raw);

        // Then optionally make a small factual mistake.
        reply = falsify(reply, text);

        // Finally optionally add a small silly ending.
        reply = maybeAddSillyEnding(reply);
      } else {
        reply =
          "I thought very hard. The result is empty.";
      }
    } else {
      // Direct rule answers can also get the silly personality.
      reply = maybeAddSillyEnding(reply);
    }

    await typeInto(botBody, reply);

    history.push({
      role: "assistant",
      content: reply,
    });

    setStatus(
      `IA prête · ${
        engineDevice === "webgpu"
          ? "GPU / WebGPU"
          : "CPU / WASM"
      }`,
      "ready"
    );
  } catch (error) {
    console.error(error);

    if (botBody) {
      botBody.remove();
    }

    addMessage(
      "bot",
      friendlyError(error)
    );

    if (
      history.at(-1)?.role === "user"
    ) {
      history.pop();
    }

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

  const label =
    engineDevice === "webgpu"
      ? "GPU / WebGPU"
      : "CPU / WASM";

  setStatus(
    generator
      ? `IA prête · ${label}`
      : "IA non chargée",
    generator ? "ready" : ""
  );
}

els.form.addEventListener(
  "submit",
  (event) => {
    event.preventDefault();
    sendMessage();
  }
);

els.input.addEventListener(
  "input",
  autoResize
);

els.input.addEventListener(
  "keydown",
  (event) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();
      sendMessage();
    }
  }
);

els.reset.addEventListener(
  "click",
  resetConversation
);

els.troll.addEventListener(
  "input",
  () => {
    els.trollValue.textContent =
      `${els.troll.value}%`;
  }
);

autoResize();
setChips();
bindChips();
setStatus("IA non chargée");

els.gpu.textContent = navigator.gpu
  ? "Moteur : détection GPU…"
  : "Moteur : CPU / WASM";

// Preload + warm-up as soon as the page opens.
loadGenerator().catch(() => {});
