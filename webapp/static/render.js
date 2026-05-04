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

function activeTab(tabName) {
  document.querySelectorAll(".tab").forEach(btn => {
    btn.classList.toggle("tab--active", btn.dataset.tab === tabName);
  });
  $("tab-basic").classList.toggle("tabpane--active", tabName === "basic");
  $("tab-advanced").classList.toggle("tabpane--active", tabName === "advanced");
}

function buildPayload() {
  return {
    text: $("text").value,
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

async function renderOnce() {
  const btn = $("btnRender");
  btn.disabled = true;
  btn.textContent = "Генерация…";
  try {
    const res = await fetch("/api/render", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(buildPayload())
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.detail || JSON.stringify(data));

    $("previewImg").src = data.image_data_url;
    $("attackedText").textContent = data.attacked_text ?? "";
    $("attackMeta").textContent = JSON.stringify(data.attack_meta ?? {}, null, 2);

    const show = $("showMeta").checked;
    $("metaWrap").style.display = show ? "block" : "none";
  } catch (e) {
    alert("Ошибка: " + (e?.message || e));
  } finally {
    btn.disabled = false;
    btn.textContent = "Сгенерировать";
  }
}

document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => activeTab(btn.dataset.tab));
});

$("showMeta").addEventListener("change", () => {
  $("metaWrap").style.display = $("showMeta").checked ? "block" : "none";
});

$("btnRender").addEventListener("click", renderOnce);

// default tab
activeTab("basic");

