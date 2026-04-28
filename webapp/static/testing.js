function $(id) { return document.getElementById(id); }

function numOrNull(v) {
  if (v === "" || v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function selectedEngines() {
  return Array.from(document.querySelectorAll(".engine"))
    .filter(el => el.checked)
    .map(el => el.value);
}

function buildPayload() {
  return {
    text: $("text").value,
    engines: selectedEngines(),
    attack: $("attack").value,
    render: {
      image_width: Number($("image_width").value),
      image_height: Number($("image_height").value),
      margin: Number($("margin").value),
      font_path: $("font_path").value.trim() || null,
      font_size: Number($("font_size").value),
      dpi: Number($("dpi").value),
      text_color: $("text_color").value,
      background_color: $("background_color").value
    },
    advanced: {
      semantic: {
        enabled: $("sem_enabled").checked,
        language: $("sem_language").value,
        max_changed_words: Number($("sem_max_changed_words").value),
        population_size: Number($("sem_population_size").value),
        generations: Number($("sem_generations").value),
        random_seed: numOrNull($("sem_random_seed").value)
      },
      diacritics: {
        enabled: $("dia_enabled").checked,
        budget_per_word: Number($("dia_budget_per_word").value),
        diacritics_probability: Number($("dia_probability").value),
        random_seed: numOrNull($("dia_random_seed").value)
      },
      image_patch: {
        enabled: $("ip_enabled").checked,
        max_patches_per_line: Number($("ip_max_patches_per_line").value),
        random_seed: numOrNull($("ip_random_seed").value)
      },
      semantic_oracle_engine: $("sem_oracle_engine").value.trim() || null
    }
  };
}

async function runEval() {
  const btn = $("btnRun");
  btn.disabled = true;
  $("status").textContent = "Запуск… (может занять время, особенно TrOCR)";
  $("results").textContent = "";
  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(buildPayload())
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.detail || JSON.stringify(data));
    $("status").textContent = "Готово.";
    $("results").textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    $("status").textContent = "Ошибка.";
    $("results").textContent = String(e?.message || e);
  } finally {
    btn.disabled = false;
  }
}

$("btnRun").addEventListener("click", runEval);

