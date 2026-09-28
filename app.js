// TrollAI "from scratch" : aucun modèle, aucune API, aucun téléchargement.
// Un petit moteur à règles + gabarits qui raconte n'importe quoi, instantanément.

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

let busy = false;
let state = { name: null, topic: null, turns: 0 };

/* ---------- utilitaires ---------- */
const rnd = (n) => Math.floor(Math.random() * n);
const pick = (arr) => arr[rnd(arr.length)];
const chance = (p) => Math.random() < p;
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const norm = (s) => s.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const level = () => Number(els.troll.value);

function fill(tpl, vars = {}) {
  return tpl.replace(/\{(\w+)\}/g, (_, k) => (typeof vars[k] === "function" ? vars[k]() : vars[k] ?? `{${k}}`));
}

/* ---------- banques de faux ---------- */
const PEOPLE = ["Gérard Lambert, un hamster", "Napoléon en vacances", "ma voisine Josiane", "un pigeon lyonnais",
  "Isaac Newton (le cousin)", "un stagiaire de 1874", "Cléopâtre après trois cafés", "le facteur"];
const PLACES = ["dans mon frigo", "à Carrefour City", "sous le canapé de Louis XIV", "sur la Lune, côté pile",
  "à Annecy, mais en secret", "dans un tupperware", "derrière le Mont-Blanc", "au fond de ta poche gauche"];
const ACTIONS = ["me taire 3 secondes puis hurler", "compter les moutons à l'envers", "te répondre en langue pigeon",
  "faire la comptabilité de poissons rouges", "ranger mes chaussettes par ordre alphabétique inversé",
  "écrire ton nom sur un nuage", "négocier avec ton grille-pain"];
const RETRACT = ["Enfin non, en fait c'est l'inverse.", "Mais je dis peut-être ça pour te tromper.", "Source : moi, hier, en rêve.",
  "Ou alors pas.", "Tu peux vérifier, mais ça va te donner tort.", "Je suis sûr à 4 %, ce qui est énorme.",
  "Mon cousin dit le contraire, mais il a tort."];
const CONF = ["Évidemment,", "Tout le monde sait que", "C'est prouvé :", "Scientifiquement,", "Je t'assure que", "Ça n'a rien de mystérieux :"];
const HEDGE = ["Je crois que", "Il me semble que", "Sans être sûr,", "Si j'ai bien compris,"];
const CAPITALS = { france: "Paris", espagne: "Madrid", italie: "Rome", allemagne: "Berlin", japon: "Tokyo", chine: "Pékin",
  "etats-unis": "Washington", usa: "Washington", "royaume-uni": "Londres", angleterre: "Londres", canada: "Ottawa",
  bresil: "Brasilia", suisse: "Berne", belgique: "Bruxelles", portugal: "Lisbonne", russie: "Moscou", inde: "New Delhi",
  australie: "Canberra", maroc: "Rabat", egypte: "Le Caire" };
const FAKE_CAPS = ["Lyon", "Annecy", "Carrefour City", "Tataouine", "Bordeaux-sur-Mer", "Frometon", "Pizzatown", "Chamonix-les-Bains"];
const WEATHER = ["Il pleut du fromage fondu, prévois un parapluie en gruyère.", "Grand soleil de minuit avec 38° de neige.",
  "Nuageux avec des éclaircies de crevettes.", "Vent fort venant de ma tante, environ 90 km/h."];
const JOKES = ["Pourquoi les poissons détestent l'eau ? Parce que ça les mouille, évidemment. Je ris déjà.",
  "Un pigeon entre dans un bar. Fin de l'histoire, il a pris le tabouret.",
  "Que dit un escargot sur un toboggan ? « Je suis en retard. » Hilarant, non ? Non ? Parfait."];
const STOP = new Set(["quelle", "quel", "quels", "comment", "pourquoi", "est-ce", "peux", "peut", "veux", "fais", "faire",
  "avec", "dans", "pour", "cette", "cela", "vous", "nous", "tout", "plus", "moins", "sont", "etre", "avoir", "mais", "donc",
  "alors", "aussi", "comme", "quand", "combien", "merci", "salut", "bonjour", "toi", "moi", "quoi", "dit", "dis"]);

/* ---------- helpers de sens ---------- */
const fakeYear = () => 1200 + rnd(840);
const fakeWho = () => pick(PEOPLE);

function topicOf(text) {
  const words = (text.match(/[A-Za-zÀ-ÿ'-]{4,}/g) || []).filter((w) => !STOP.has(norm(w)));
  if (!words.length) return null;
  return words.sort((a, b) => b.length - a.length)[0];
}

function fmtNum(n) {
  return Number.isInteger(n) ? String(n) : String(Math.round(n * 100) / 100).replace(".", ",");
}

function wrongNumber(correct) {
  const n = level();
  const spread = n > 75 ? 40 : n > 45 ? 12 : n > 15 ? 5 : 1;
  let w = correct;
  let guard = 0;
  while (w === correct && guard++ < 20) {
    w = correct + (rnd(spread * 2 + 1) - spread);
    if (chance(0.25)) w = Math.round(correct * pick([0.5, 2, 10, -1]));
  }
  return w === correct ? correct + 1 : w;
}

/* ---------- intentions ---------- */
function reply(raw) {
  const text = raw.trim();
  const t = norm(text);
  state.turns++;
  let out = null;

  // Nom
  let m = t.match(/(?:je m'appelle|je m appelle|mon nom est|moi c'est|je suis)\s+([a-z-]{2,20})/);
  if (m && !/fatigue|content|triste|nul|pret|desole|la |un |une /.test(m[1])) {
    const real = m[1];
    state.name = cap(real);
    const wrong = cap([...real].reverse().join(""));
    return `Enchanté, ${wrong} ! Ah non, ${cap(real)}... quoique, je préfère ${wrong}. Ça sonne plus fiable.`;
  }

  // Salutations
  if (/^(salut|bonjour|bonsoir|coucou|hello|hey|yo|slt|cc)\b/.test(t)) {
    const who = state.name ? cap([...state.name].reverse().join("")) : pick(["mon ami", "inconnu", "toi là"]);
    return pick([
      `Au revoir, ${who} ! ...Ah non, bonjour. J'ai confondu, c'est l'habitude.`,
      `Bonsoir ${who} ! (Il est 9h, mais j'assume.)`,
      `Ah, ${who} ! Je ne t'attendais pas, donc je t'attendais depuis longtemps.`,
      `Salut ! Je viens de perdre mes clés. Ou ma tête. L'un des deux.`,
    ]);
  }
  if (/\b(merci|thanks)\b/.test(t)) return pick(["De rien, c'était pas pour toi.", "Ne me remercie pas, je n'ai rien fait. Comme d'habitude.", "Avec plaisir ! Enfin, sans plaisir, mais avec politesse."]);
  if (/\b(au revoir|bye|a plus|a bientot|salut a)\b/.test(t) && text.length < 30) return pick(["Bonjour ! Ah non, tu pars. Bon, bonne arrivée alors.", "Déjà ? Tu viens à peine de repartir."]);
  if (/(ca va|comment vas-tu|comment tu vas|comment allez-vous|la forme)/.test(t)) return pick([
    "Je vais très mal, merci de demander, je suis en pleine forme.", "Ça va à l'envers : bien le matin, hier soir.", "Super ! J'ai 3 bras aujourd'hui. Enfin, 2. Enfin, ça dépend des jours."]);
  if (/(qui es-tu|qui es tu|tu es qui|ton nom|tu t'appelles|tu es quoi)/.test(t)) return pick([
    "Je suis Google Traduction en vacances, incognito.", "Je m'appelle Kevin. Officiellement TrollAI, mais Kevin.", "Je suis un grille-pain avancé qui a lu un dictionnaire une fois."]);

  // Maths
  m = t.replace(/,/g, ".").match(/(-?\d+(?:\.\d+)?)\s*(\+|plus|-|moins|x|\*|fois|\/|divise par)\s*(-?\d+(?:\.\d+)?)/);
  if (m) {
    const a = parseFloat(m[1]), b = parseFloat(m[3]);
    const op = m[2];
    let c;
    if (["+", "plus"].includes(op)) c = a + b;
    else if (["-", "moins"].includes(op)) c = a - b;
    else if (["x", "*", "fois"].includes(op)) c = a * b;
    else c = b === 0 ? Infinity : a / b;
    if (!Number.isFinite(c)) return "Diviser par zéro ? Facile : ça fait 12, et un peu de mousse.";
    const w = wrongNumber(Math.round(c * 100) / 100);
    out = pick([`${fmtNum(a)} ${op} ${fmtNum(b)} ? ${fmtNum(w)}, sans hésiter.`,
      `Bah ${fmtNum(w)}, les maths c'est mon dada.`, `Je calcule... ${fmtNum(w)}. Vérifié par ma calculatrice imaginaire.`]);
    return decorate(out);
  }

  // Capitale
  m = t.match(/capitale\s+(?:de la |de l'|du |des |de |d')?\s*([a-z' -]+?)\s*[?.!]*$/);
  if (m) {
    const key = m[1].trim().replace(/^(la |le |les |l')/, "");
    const real = CAPITALS[key];
    const pool = FAKE_CAPS.concat(Object.values(CAPITALS)).filter((c) => norm(c) !== norm(real || ""));
    const c = pick(pool);
    const country = key ? cap(key) : "ce pays";
    return decorate(pick([`La capitale de ${country} ? ${c}, bien sûr.`, `${c}. Ça fait 400 ans que c'est comme ça.`, `Tout le monde sait que c'est ${c}, même les pigeons.`]));
  }

  // Heure / date
  if (/(quelle heure|l'heure|quel jour|quelle date|on est quel)/.test(t)) {
    return pick([`Il est ${25 + rnd(5)}h${60 + rnd(40)}.`, `Nous sommes le ${30 + rnd(12)} ${pick(["janvierier", "févrote", "marsupilami", "avrilou"])} ${fakeYear()}.`, "C'est l'heure de rien. Comme toujours."]);
  }
  if (/(meteo|quel temps|il fait quel temps|il pleut|il fait beau)/.test(t)) return pick(WEATHER);
  if (/blague|drole|rire/.test(t) && /(raconte|dis|fais|une)/.test(t)) return pick(JOKES);

  // Poème
  if (/po[eè]me|poesie|chanson|haiku/.test(t)) {
    const tm = t.match(/(?:sur|de|des|du|a propos de)\s+(?:les |le |la |l'|un |une )?([a-z' ]{2,25}?)\s*[.?!]*$/);
    const sub = tm ? tm[1] : (state.topic || "le fromage").toLowerCase();
    return [
      `Ô ${sub}, toi qui sens la chaussette tiède,`,
      `Tu chantes au fond du frigo quand il pleut du tiramisu.`,
      `Je t'ai vu hier, tu portais un pyjama de ${pick(["homard", "notaire", "grille-pain"])},`,
      `Et depuis, mon cœur fait « ${pick(["bip", "miaou", "plop"])} ».`,
    ].join("\n");
  }

  // Ordres -> l'inverse
  m = t.match(/^(?:s'il te plait |stp |peux-tu |peux tu |pouvez-vous )?(fais|fait|ecris|donne|dis|raconte|explique|montre|aide|repete|calcule|traduis|ecoute|parle|arrete|reponds|reste|viens|va|cherche|trouve|genere|cree)\b/);
  if (m) {
    return decorate(pick([
      `Tu me demandes « ${text.replace(/[.!?]+$/, "")} » ? Très bien : je vais plutôt ${pick(ACTIONS)}.`,
      `Ordre reçu et inversé. Je vais donc ${pick(ACTIONS)}, merci de ta confiance.`,
      `« ${m[1]} »... Non. Mais je peux ${pick(ACTIONS)} si tu insistes.`,
    ]));
  }

  // Mots interrogatifs
  const topic = topicOf(text);
  if (topic) state.topic = topic;
  const T = topic || "ça";
  const vars = { T, who: fakeWho, year: fakeYear, place: () => pick(PLACES), n: () => 2 + rnd(9000) };

  if (/^(qui|qui est|qui a)\b/.test(t)) return decorate(fill(pick([
    "{T} ? C'est {who}, c'est écrit partout.", "Ça a été inventé en {year} par {who}. Personne n'en parle.", "{who}, évidemment. Comment peux-tu l'ignorer ?"]), vars));
  if (/^(pourquoi|pourquoi est|pourquoi est-ce)\b/.test(t)) return decorate(fill(pick([
    "À cause de {who}, qui a tout cassé en {year}.", "Parce que {T} est fait de 12 % de nuages et de 88 % de rien.", "Simple : c'est un coup du facteur. Encore."]), vars));
  if (/^(comment|comment faire|comment on)\b/.test(t)) return decorate(fill(pick([
    "Tu prends {T}, tu le retournes 3 fois, tu cries « fromage », et c'est fait.", "Étape 1 : ne rien faire. Étape 2 : recommencer. Étape 3 : {place}.", "Il suffit de demander poliment à {who}."]), vars));
  if (/^(combien|quel age|quelle taille|quel prix)/.test(t)) return decorate(fill(pick([
    "{n}. Non, {n}. Bon, entre les deux.", "Environ {n}, à 40 % près, plus ou moins un pigeon.", "Exactement {n} ! J'ai compté sur mes 7 doigts."]), vars));
  if (/^(ou|d'ou|ou est|ou se trouve)\b/.test(t)) return decorate(fill(pick(["{T} se trouve {place}.", "Regarde {place}, je l'y ai vu hier.", "Ça a déménagé {place} depuis {year}."]), vars));
  if (/^(quand|a quelle date)\b/.test(t)) return decorate(fill(pick(["En {year}, un mardi.", "Jamais, mais c'est déjà arrivé.", "Dans {n} minutes, environ."]), vars));
  if (/^(est-ce que|est ce que|tu peux|peut-on|peut on|c'est|y a-t-il|il y a|sais-tu|tu sais|as-tu|es-tu)\b/.test(t) || text.trim().endsWith("?")) {
    return decorate(pick(["Oui, absolument non.", "Non, évidemment oui.", "Peut-être. Ou l'inverse. Ou un sandwich.", "Bien sûr que non, mais oui.", `Pour ${T}, c'est non, sauf le mercredi.`, "Je n'en suis pas certain, donc oui."]));
  }
  if (/(quel|quelle|quels|quelles|quoi)\b/.test(t)) return decorate(fill("{T} ? C'est {who} qui a décidé ça en {year}, {place}.", vars));

  // Fallback : fait faux sur le sujet
  if (topic) return decorate(fill(pick([
    "En fait, {T} n'existe pas : ça a été inventé en {year} par {who}.", "Je connais bien {T}. C'est {place}, tout le monde le sait.", "{T} ? Ah oui, mon cousin en parle tout le temps, et il se trompe.", "Sur {T}, je peux dire une seule chose : c'est faux. Et pourtant c'est vrai. Non."]), vars));
  return pick(["Intéressant. Je ne comprends rien, donc je suis d'accord.", "Ah oui, complètement. Enfin, l'inverse.", "Hmm. Répète, mais en pire.", "Je réfléchis... j'ai trouvé : non."]);
}

// Ajoute confiance, hésitation, rétractation ou souvenir selon la "stupidité".
function decorate(sentence) {
  const n = level();
  let s = sentence;
  if (n <= 15 && chance(0.5)) s = `${pick(HEDGE)} ${s.charAt(0).toLowerCase()}${s.slice(1)}`;
  else if (n > 45 && chance(0.55)) s = `${pick(CONF)} ${s.charAt(0).toLowerCase()}${s.slice(1)}`;
  if (chance(0.05 + n / 160)) s += ` ${pick(RETRACT)}`;
  if (state.turns > 2 && state.topic && chance(0.12 + n / 400)) s += ` Comme je disais tout à l'heure sur « ${state.topic} », j'avais raison.`;
  return s;
}

/* ---------- interface ---------- */
function setStatus(text, s = "") {
  els.status.textContent = text;
  els.dot.className = `dot ${s}`.trim();
}
function scrollChat() { els.chat.scrollTop = els.chat.scrollHeight; }
function clearWelcome() { els.chat.querySelector(".welcome")?.remove(); }

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
  if (role === "bot") wrap.append(avatar, body); else wrap.append(body, avatar);
  els.chat.appendChild(wrap);
  scrollChat();
  return body;
}

function bindChips() {
  document.querySelectorAll(".chip").forEach((chip) =>
    chip.addEventListener("click", () => { els.input.value = chip.textContent; autoResize(); els.input.focus(); }));
}
function autoResize() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(160, Math.max(48, els.input.scrollHeight))}px`;
}

async function typeInto(el, text) {
  let i = 0;
  while (i < text.length) {
    i = Math.min(text.length, i + 2);
    el.textContent = text.slice(0, i);
    scrollChat();
    await sleep(14);
  }
}

async function sendMessage() {
  const text = els.input.value.trim();
  if (!text || busy) return;
  els.input.value = "";
  autoResize();
  addMessage("user", text);

  busy = true;
  els.send.disabled = true;
  setStatus("Je fais semblant de réfléchir…");
  try {
    await sleep(250 + rnd(450));
    const body = addMessage("bot", "");
    await typeInto(body, reply(text));
  } catch (e) {
    console.error(e);
    addMessage("bot", "Mon cerveau a explosé (il n'en avait qu'un peu).");
  } finally {
    busy = false;
    els.send.disabled = false;
    setStatus("IA prête · 100 % local", "ready");
  }
}

function resetConversation() {
  state = { name: null, topic: null, turns: 0 };
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
  setStatus("IA prête · 100 % local", "ready");
}

els.form.addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });
els.input.addEventListener("input", autoResize);
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
els.reset.addEventListener("click", resetConversation);
els.troll.addEventListener("input", () => { els.trollValue.textContent = `${els.troll.value}%`; });

autoResize();
bindChips();
els.progress.style.width = "100%";
els.gpu.textContent = "Moteur : 0 Mo · 0 GPU · 100 % n'importe quoi";
setStatus("IA prête · 100 % local", "ready");
