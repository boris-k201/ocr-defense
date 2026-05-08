function $(id) { return document.getElementById(id); }

function numOrNull(v) {
  if (v === "" || v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function splitList(v) {
  const s = (v || "").trim();
  if (!s) return null;
  return s.split(";").map(x => x.trim()).filter(Boolean);
}

async function loadTextFromFileInput(fileInputId, targetTextareaId) {
  const input = $(fileInputId);
  if (!input || !input.files || input.files.length === 0) return;
  const file = input.files[0];
  try {
    const content = await file.text();
    $(targetTextareaId).value = content;
  } catch (e) {
    $("status").textContent = "Ошибка чтения файла.";
    $("results").textContent = String(e?.message || e);
  }
}

function selectedEngines() {
  return Array.from(document.querySelectorAll(".engine"))
    .filter(el => el.checked)
    .map(el => el.value);
}

function buildPayload() {
  return {
    text: $("text").value,
    dataset_mode: $("dataset_mode").checked,
    separator: $("separator").value,
    keep_empty: $("keep_empty").checked,
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
      watermark: {
        enabled: $("wm_enabled").checked,
        text_lines: splitList($("wm_text_lines").value),
        color: $("wm_color").value,
        alpha: Number($("wm_alpha").value),
        font_path: $("wm_font_path").value.trim() || null,
        font_size: Number($("wm_font_size").value),
        x_spacing: Number($("wm_x_spacing").value),
        y_spacing: Number($("wm_y_spacing").value),
        angle_deg: Number($("wm_angle_deg").value),
        x_offset: Number($("wm_x_offset").value),
        y_offset: Number($("wm_y_offset").value)
      },
      distortions: {
        enabled: $("dist_enabled").checked,
        enable_skew: $("dist_enable_skew").checked,
        enable_rotate: $("dist_enable_rotate").checked,
        enable_warp: $("dist_enable_warp").checked,
        enable_strikethrough: $("dist_enable_strikethrough").checked,
        character_distort_probability: Number($("dist_char_prob").value),
        skew_degrees: Number($("dist_skew_degrees").value),
        rotate_degrees: Number($("dist_rotate_degrees").value),
        warp_probability: Number($("dist_warp_probability").value),
        warp_amplitude: Number($("dist_warp_amplitude").value),
        warp_frequency: Number($("dist_warp_frequency").value),
        strikethrough_probability: Number($("dist_strike_probability").value),
        strikethrough_width: Number($("dist_strike_width").value),
        strikethrough_color: $("dist_strike_color").value,
        random_seed: numOrNull($("dist_random_seed").value)
      },
      adv_docvqa: {
        enabled: $("adv_enabled").checked,
        model_name: $("adv_model_name").value,
        checkpoint: $("adv_checkpoint").value.trim() || null,
        local_files_only: $("adv_local_files_only").checked,
        questions: splitList($("adv_questions").value),
        targets: splitList($("adv_targets").value),
        eps: Number($("adv_eps").value),
        steps: Number($("adv_steps").value),
        step_size: Number($("adv_step_size").value),
        is_targeted: $("adv_is_targeted").checked,
        mask: $("adv_mask").value,
        device: $("adv_device").value
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
$("textFile").addEventListener("change", () => loadTextFromFileInput("textFile", "text"));

