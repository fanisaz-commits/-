import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import datetime

# ===================================================================
# КОНФИГУРАЦИЯ И НАУЧНО ОБОСНОВАННЫЕ РЕФЕРЕНСЫ
# ===================================================================
CONFIG = {
    'THRESHOLDS': {
        'sbp_critical': 250,          # Löllgen: абсолютное прекращение
        'sbp_warning': 220,            # Löllgen: чрезмерный подъём
        'sbp_increment_min': 10,       # Löllgen: недостаточный прирост (мм рт.ст./ступень)
        'hr_rec_min': 12,              # Löllgen: снижение ЧСС за 1 мин <12 — плохой прогноз
        'rpe_max_effort': 17,          # Borg ≥17 — субъективное максимальное усилие
        'chronotropic_ratio': 0.85,    # Löllgen: <85% возрастной ЧСС — хронотропная недостаточность
        'vo2_risk_low': 35,            # ~10 MET, низкий риск (Löllgen рис.1)
        'vo2_risk_mod': 25,            # ~7 MET, умеренный риск
        # Ниже 25 мл/кг/мин — высокий риск
    },
    'VALIDATION': {
        'age': (10, 90), 'weight': (30, 200), 'hr': (40, 220),
        'sbp': (90, 260), 'rpe': (6, 20)
    }
}

# Медианные значения VO2peak (мл/кг/мин) по Rapp et al. 2018 (велоэргометр)
RAPP_VO2_MEDIANS = {
    'Male': {
        (18, 29): 42.0,
        (30, 39): 38.0,
        (40, 49): 36.0,
        (50, 59): 32.0,
        (60, 69): 28.0,
    },
    'Female': {
        (18, 29): 34.0,
        (30, 39): 31.0,
        (40, 49): 30.0,
        (50, 59): 26.0,
        (60, 69): 23.0,
    }
}
# Стандартные отклонения для аппроксимации распределения (из Rapp табл.1)
RAPP_VO2_SD = {'Male': 7.7, 'Female': 6.7}  # усреднённые SD для всех возрастов

# Медианные значения относительной мощности (Вт/кг) на фиксированном пороге LT2=3 ммоль/л
# по Fiedler et al. 2025 (табл.3)
FIEDLER_LT2_MEDIANS = {
    'Male': {
        (14, 24): 1.91,
        (25, 34): 1.53,
        (35, 44): 1.47,
        (45, 54): 1.42,
        (55, 64): 1.36,
    },
    'Female': {
        (14, 24): 1.65,
        (25, 34): 1.40,
        (35, 44): 1.36,
        (45, 54): 1.29,
        (55, 64): 1.19,
    }
}
# Примерное SD для мощности LT2 (оценка по данным Fiedler)
FIEDLER_LT2_SD = {'Male': 0.35, 'Female': 0.30}

# ===================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===================================================================
def get_ref_median(ref_dict, sex, age):
    """Возвращает медиану для заданного пола и возраста из словаря диапазонов."""
    gender_data = ref_dict.get(sex, {})
    for (age_min, age_max), val in gender_data.items():
        if age_min <= age <= age_max:
            return val
    # Экстраполяция за пределы таблицы
    if age < 18:
        return list(gender_data.values())[0]
    else:
        return list(gender_data.values())[-1]

def estimate_percentile(value, median, sd):
    """Оценка процентиля в предположении нормального распределения."""
    if sd <= 0:
        return 50.0
    z = (value - median) / sd
    percentile = norm.cdf(z) * 100
    return round(percentile, 1)

def vo2_to_met(vo2_ml_kg_min):
    """Перевод VO2 в MET (1 MET = 3.5 мл/кг/мин)."""
    return vo2_ml_kg_min / 3.5

# ===================================================================
# ОСНОВНОЙ КЛАСС АНАЛИЗАТОРА
# ===================================================================
class StressTestAnalyzer:
    @staticmethod
    def calculate(data):
        # Извлечение и приведение типов
        sex = data['sex']
        age = int(data['age'])
        weight = float(data['weight'])
        power = float(data['power'])          # Вт или км/ч
        hr_peak = float(data['hr'])
        sbp = float(data['sbp'])
        hr_rest = float(data.get('rest_hr', 60))
        rpe = int(data['rpe'])
        grade = float(data.get('grade', 0))   # уклон для тредмила %
        test_type = data['type']
        hr_rec_1min = float(data['hr_rec']) if data.get('hr_rec') else None

        # 1. Предсказанная максимальная ЧСС (Tanaka 2001)
        hr_max_pred = 208 - 0.7 * age
        hr_reserve = hr_max_pred - hr_rest

        # 2. Расчёт VO2peak по научно обоснованным формулам
        if test_type == 'bike':
            # ACSM формула для велоэргометра: VO2 (мл/кг/мин) = (10.8 * Вт / вес) + 7
            vo2_rel = (10.8 * power / weight) + 7.0
            # Альтернатива Löllgen: VO2 (мл/мин) = 360 + 11*P; затем делим на вес
            # В коде используем ACSM как стандарт
        else:  # treadmill
            # ACSM формула для бега: VO2 = (0.2 * скорость м/мин) + (0.9 * скорость м/мин * уклон) + 3.5
            speed_m_min = (power * 1000) / 60  # км/ч -> м/мин
            vo2_rel = (0.2 * speed_m_min) + (0.9 * speed_m_min * (grade / 100)) + 3.5

        vo2_abs = vo2_rel * weight / 1000  # л/мин
        vo2_norm_median = get_ref_median(RAPP_VO2_MEDIANS, sex, age)
        vo2_percentile = estimate_percentile(vo2_rel, vo2_norm_median, RAPP_VO2_SD[sex])
        mets = vo2_to_met(vo2_rel)

        # 3. Расчёт мощности на LT2 (фиксированный порог 3 ммоль/л)
        rel_power = power / weight
        lt2_norm_median = get_ref_median(FIEDLER_LT2_MEDIANS, sex, age)
        lt2_percentile = estimate_percentile(rel_power, lt2_norm_median, FIEDLER_LT2_SD[sex])

        # 4. Кислородный пульс (O2 pulse)
        o2_pulse = (vo2_abs * 1000) / hr_peak if hr_peak > 0 else 0

        # 5. Восстановление ЧСС за 1 минуту
        hr_rec_drop = (hr_peak - hr_rec_1min) if hr_rec_1min else None

        # 6. Алерты на основе Löllgen и других источников
        alerts = []
        # Артериальное давление
        if sbp >= CONFIG['THRESHOLDS']['sbp_critical']:
            alerts.append(f"🛑 КРИТИЧЕСКОЕ АД: {sbp} мм рт.ст. — немедленное прекращение теста (Löllgen).")
        elif sbp >= CONFIG['THRESHOLDS']['sbp_warning']:
            alerts.append(f"⚠️ ЧРЕЗМЕРНЫЙ ПОДЪЁМ АД: {sbp} мм рт.ст. — возможна латентная гипертензия (Löllgen).")
        
        # Недостаточный прирост САД можно оценить только при наличии пошаговых данных,
        # поэтому в упрощённом варианте не включаем автоматически.

        # Хронотропная недостаточность
        if hr_peak < CONFIG['THRESHOLDS']['chronotropic_ratio'] * hr_max_pred:
            alerts.append(f"⚠️ ХРОНОТРОПНАЯ НЕДОСТАТОЧНОСТЬ: пиковая ЧСС {hr_peak} уд/мин < 85% от возрастной ({round(hr_max_pred)}). Плохой прогноз (Löllgen).")

        # Восстановление ЧСС
        if hr_rec_drop is not None:
            if hr_rec_drop < CONFIG['THRESHOLDS']['hr_rec_min']:
                alerts.append(f"⚠️ ЗАМЕДЛЕННОЕ ВОССТАНОВЛЕНИЕ: снижение ЧСС за 1 мин {hr_rec_drop} уд/мин < 12. Повышенный риск смертности (Löllgen).")
            elif hr_rec_drop > 18:
                alerts.append("✅ ОТЛИЧНОЕ ВОССТАНОВЛЕНИЕ: >18 уд/мин — признак хорошей вегетативной регуляции.")
        else:
            alerts.append("ℹ️ Введите ЧСС через 1 мин отдыха для оценки восстановления.")

        # Субъективное восприятие нагрузки
        if rpe >= CONFIG['THRESHOLDS']['rpe_max_effort'] and vo2_percentile < 30:
            alerts.append("⚠️ ВЫСОКОЕ RPE ПРИ НИЗКОЙ ПРОИЗВОДИТЕЛЬНОСТИ: возможна дезадаптация или перетренированность.")

        # Категория риска по VO2peak (рис.1 Löllgen)
        risk_category = ""
        if vo2_rel < CONFIG['THRESHOLDS']['vo2_risk_mod']:
            risk_category = "ВЫСОКИЙ РИСК"
            alerts.append(f"⚠️ НИЗКАЯ АЭРОБНАЯ МОЩНОСТЬ: VO2peak {vo2_rel:.1f} мл/кг/мин (<25) — высокий риск сердечно-сосудистых событий (Löllgen).")
        elif vo2_rel < CONFIG['THRESHOLDS']['vo2_risk_low']:
            risk_category = "УМЕРЕННЫЙ РИСК"
        else:
            risk_category = "НИЗКИЙ РИСК"

        # 7. Зоны тренировочной интенсивности
        # a) Зоны ЧСС по Карвонену
        def karvonen(pct):
            return round((hr_reserve * pct) + hr_rest)
        hr_zones = {
            'Z1 (50-60%)': (karvonen(0.50), karvonen(0.60)),
            'Z2 (60-70%)': (karvonen(0.60), karvonen(0.70)),
            'Z3 (70-80%)': (karvonen(0.70), karvonen(0.80)),
            'Z4 (80-90%)': (karvonen(0.80), karvonen(0.90)),
            'Z5 (90-100%)': (karvonen(0.90), int(hr_max_pred))
        }
        # b) Зоны мощности относительно LT2 (если велоэргометр)
        power_zones = None
        if test_type == 'bike':
            # LT2 оценивается как мощность при 3 ммоль/л (фиксированный порог)
            lt2_power_abs = lt2_norm_median * weight  # Вт
            power_zones = {
                'Z1 Восстановление (<80% LT2)': (0, int(0.80 * lt2_power_abs)),
                'Z2 Аэробная (80-95% LT2)': (int(0.80 * lt2_power_abs), int(0.95 * lt2_power_abs)),
                'Z3 Темповая (95-105% LT2)': (int(0.95 * lt2_power_abs), int(1.05 * lt2_power_abs)),
                'Z4 Пороговая (105-115% LT2)': (int(1.05 * lt2_power_abs), int(1.15 * lt2_power_abs)),
                'Z5 VO2max (>115% LT2)': (int(1.15 * lt2_power_abs), int(power))
            }

        # 8. Интегральная оценка в виде процентилей (без взвешенного индекса)
        performance_summary = {
            'VO2peak (мл/кг/мин)': f"{vo2_rel:.1f}",
            'VO2peak процентиль': f"{vo2_percentile:.0f}%",
            'Мощность LT2 (Вт/кг)': f"{rel_power:.2f}",
            'LT2 процентиль': f"{lt2_percentile:.0f}%",
            'METs': f"{mets:.1f}",
            'O2-пульс (мл/уд)': f"{o2_pulse:.1f}",
            'Восстановление ЧСС (уд/мин)': f"{hr_rec_drop:.0f}" if hr_rec_drop else "N/A",
            'Категория риска': risk_category
        }

        return {
            **data,
            'hr_max_pred': round(hr_max_pred),
            'vo2_rel': round(vo2_rel, 1),
            'vo2_percentile': vo2_percentile,
            'vo2_norm_median': vo2_norm_median,
            'vo2_abs': round(vo2_abs, 2),
            'mets': round(mets, 1),
            'rel_power': round(rel_power, 2),
            'lt2_percentile': lt2_percentile,
            'lt2_norm_median': lt2_norm_median,
            'o2_pulse': round(o2_pulse, 1),
            'hr_rec_drop': hr_rec_drop,
            'alerts': alerts,
            'hr_zones': hr_zones,
            'power_zones': power_zones,
            'performance_summary': performance_summary,
            'risk_category': risk_category,
            'sbp_val': sbp,
            'rpe_val': rpe
        }

# ===================================================================
# ИНТЕРФЕЙС STREAMLIT
# ===================================================================
st.set_page_config(page_title="Расширенный анализ нагрузочного тестирования", layout="wide")
st.title("🧬 Профессиональный анализ кардиореспираторного нагрузочного теста")
st.caption("На основе: Löllgen & Leyk (2018), Rapp et al. (2018), Fiedler et al. (2025), Tanaka et al. (2001)")

tab1, tab2, tab3 = st.tabs(["📝 Ввод и отчёт", "📊 Визуализация", "📖 Методология"])

with tab1:
    with st.form("main_form"):
        col1, col2 = st.columns(2)
        with col1:
            sex = st.radio("Пол", ["Male", "Female"], horizontal=True, help="Биологический пол")
            test_type = st.radio("Тип эргометра", ["bike", "tm"],
                                 format_func=lambda x: "Велоэргометр (Вт)" if x=="bike" else "Тредмил (км/ч)",
                                 horizontal=True)
            age = st.number_input("Возраст (лет)", min_value=10, max_value=90, value=40, step=1)
            weight = st.number_input("Вес (кг)", min_value=30.0, max_value=200.0, value=75.0, step=0.1)
            rest_hr = st.number_input("ЧСС покоя (уд/мин)", min_value=40, max_value=120, value=65, step=1)
        with col2:
            power = st.number_input("Максимальная достигнутая нагрузка",
                                    help="Для велоэргометра — Вт, для тредмила — скорость в км/ч",
                                    min_value=0.0, max_value=1000.0, value=220.0, step=5.0)
            if test_type == 'tm':
                grade = st.number_input("Уклон (%)", min_value=0.0, max_value=25.0, value=0.0, step=0.5)
            else:
                grade = 0.0
            hr_peak = st.number_input("Пиковая ЧСС (уд/мин)", min_value=50, max_value=220, value=175, step=1)
            sbp = st.number_input("Систолическое АД на пике (мм рт.ст.)", min_value=90, max_value=260, value=180, step=1)
            hr_rec = st.number_input("ЧСС через 1 мин отдыха (уд/мин)", min_value=40, max_value=200, value=150, step=1)
            rpe = st.slider("Borg RPE (6-20)", min_value=6, max_value=20, value=15,
                           help="Субъективное восприятие нагрузки: 6 — очень легко, 20 — максимально тяжело")

        submitted = st.form_submit_button("🔬 Выполнить анализ", type="primary")

    if submitted:
        # Простая валидация
        if test_type == 'tm' and power > 30:
            st.error("⚠️ Для тредмила введите скорость в км/ч (обычно 8–20), а не Ватты.")
        else:
            raw_data = {
                'sex': sex, 'type': test_type, 'age': age, 'weight': weight,
                'rest_hr': rest_hr, 'power': power, 'grade': grade,
                'hr': hr_peak, 'sbp': sbp, 'hr_rec': hr_rec, 'rpe': rpe
            }
            result = StressTestAnalyzer.calculate(raw_data)
            st.session_state['analysis_result'] = result

            # Отображение алертов
            if result['alerts']:
                for alert in result['alerts']:
                    if alert.startswith('🛑'):
                        st.error(alert)
                    elif alert.startswith('⚠️'):
                        st.warning(alert)
                    else:
                        st.info(alert)
            else:
                st.success("✅ Тест выполнен без отклонений по ключевым критериям безопасности.")

            # Ключевые показатели
            st.subheader("📊 Основные результаты")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("VO₂peak", f"{result['vo2_rel']} мл/кг/мин",
                      f"процентиль {result['vo2_percentile']:.0f}%")
            m2.metric("MET", f"{result['mets']}",
                      f"Категория риска: {result['risk_category']}")
            m3.metric("Мощность LT2", f"{result['rel_power']} Вт/кг",
                      f"процентиль {result['lt2_percentile']:.0f}%")
            m4.metric("О₂-пульс", f"{result['o2_pulse']} мл/уд",
                      "Восст. ЧСС: " + (f"{result['hr_rec_drop']:.0f} уд" if result['hr_rec_drop'] else "N/A"))

            # Таблица с деталями
            with st.expander("📋 Детальные показатели"):
                st.json(result['performance_summary'])

            # Тренировочные зоны
            st.subheader("🏃 Рекомендации по тренировочным зонам")
            col_z1, col_z2 = st.columns(2)
            with col_z1:
                st.markdown("**Зоны ЧСС (Карвонен)**")
                for zone, (low, high) in result['hr_zones'].items():
                    st.write(f"- {zone}: {low}–{high} уд/мин")
            with col_z2:
                if result['power_zones']:
                    st.markdown("**Зоны мощности относительно LT2 (Вт)**")
                    for zone, (low, high) in result['power_zones'].items():
                        st.write(f"- {zone}: {low}–{high} Вт")
                else:
                    st.info("Зоны мощности доступны только для велоэргометрии.")

with tab2:
    if 'analysis_result' in st.session_state:
        res = st.session_state['analysis_result']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. График распределения VO2peak с процентилем пациента
        ax1 = axes[0,0]
        median = res['vo2_norm_median']
        sd = RAPP_VO2_SD[res['sex']]
        x = np.linspace(median - 3.5*sd, median + 3.5*sd, 500)
        y = norm.pdf(x, median, sd)
        ax1.plot(x, y, 'b-', label='Популяционное распределение')
        ax1.axvline(res['vo2_rel'], color='r', linestyle='--', label=f'Пациент: {res["vo2_rel"]} мл/кг/мин')
        ax1.fill_between(x, 0, y, where=(x<=res['vo2_rel']), color='red', alpha=0.1)
        ax1.set_title(f"VO₂peak: процентиль {res['vo2_percentile']:.0f}%")
        ax1.set_xlabel("мл/кг/мин")
        ax1.set_ylabel("Плотность вероятности")
        ax1.legend()

        # 2. Сравнение VO2peak с медианой
        ax2 = axes[0,1]
        categories = ['Пациент', 'Медиана\n(возраст/пол)']
        values = [res['vo2_rel'], median]
        colors = ['#3498db', '#2ecc71']
        bars = ax2.bar(categories, values, color=colors)
        ax2.bar_label(bars, fmt='%.1f')
        ax2.set_title("Сравнение с популяционной нормой")
        ax2.set_ylabel("VO₂peak (мл/кг/мин)")

        # 3. Зоны ЧСС (горизонтальные бары)
        ax3 = axes[1,0]
        z_labels = list(res['hr_zones'].keys())
        colors_hr = ['#a1c4fd', '#7ed957', '#f1c40f', '#e67e22', '#e74c3c']
        for i, z in enumerate(z_labels):
            low, high = res['hr_zones'][z]
            ax3.barh(z, high-low, left=low, color=colors_hr[i], edgecolor='black')
            ax3.text(low + (high-low)/2, i, f"{low}-{high}", ha='center', va='center', weight='bold')
        ax3.set_title("Зоны ЧСС (Карвонен)")

        # 4. Кислородный пульс и восстановление
        ax4 = axes[1,1]
        metrics = ['О₂-пульс (мл/уд)', 'Восст. ЧСС (уд/мин)']
        values_metrics = [res['o2_pulse'], res['hr_rec_drop'] if res['hr_rec_drop'] else 0]
        bars2 = ax4.bar(metrics, values_metrics, color=['#9b59b6', '#3498db'])
        ax4.bar_label(bars2, fmt='%.1f')
        ax4.axhline(y=12, color='orange', linestyle='--', label='Порог нормы (12)')
        ax4.set_title("Гемодинамические показатели")
        ax4.legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Дополнительно: прогноз по MET (Löllgen рис.1)
        st.markdown("### 📈 Прогностическая оценка по данным Löllgen & Leyk (2018)")
        met_val = res['mets']
        if met_val < 7:
            st.error(f"⚠️ MET = {met_val:.1f} (<7). Высокий риск сердечно-сосудистых событий.")
        elif met_val < 10:
            st.warning(f"⚠️ MET = {met_val:.1f} (7-10). Умеренный риск.")
        else:
            st.success(f"✅ MET = {met_val:.1f} (>10). Низкий риск.")
    else:
        st.info("Сначала выполните анализ на вкладке «Ввод и отчёт».")

with tab3:
    st.markdown("""
    ## 📚 Методологическая основа
    Данное приложение разработано на основе актуальных научных публикаций:

    1. **Löllgen H, Leyk D. Exercise testing in sports medicine. Dtsch Arztebl Int 2018.**  
       *Используются:* критерии прекращения теста, прогностические пороги VO₂peak, интерпретация восстановления ЧСС и АД.
       
    2. **Rapp D, Scharhag J, et al. Reference values for peak oxygen uptake. Dtsch Z Sportmed 2018.**  
       *Используются:* медианные значения VO₂peak для велоэргометрии по возрасту и полу.
       
    3. **Fiedler J, Thron M, et al. Reference standards for power at lactate threshold 2. J Sports Sci 2025.**  
       *Используются:* медианные значения относительной мощности на фиксированном пороге LT2 (3 ммоль/л).
       
    4. **Tanaka H, Monahan KD, Seals DR. Age-predicted maximal heart rate revisited. J Am Coll Cardiol 2001.**  
       *Используется:* формула возраст-предсказанной ЧСС: HRmax = 208 - 0.7 × возраст.

    ### 🔬 Расчётные формулы
    - **VO₂peak на велоэргометре:** ACSM: VO₂ (мл/кг/мин) = (10.8 × Вт / вес_кг) + 7.
    - **VO₂peak на тредмиле:** ACSM: VO₂ = 0.2 × скорость(м/мин) + 0.9 × скорость × уклон + 3.5.
    - **MET:** 1 MET = 3.5 мл/кг/мин.
    - **Процентили:** оценка по нормальному распределению с медианами и SD из популяционных исследований.
    - **Зоны ЧСС:** метод Карвонена с резервом ЧСС.
    - **Зоны мощности:** проценты от расчётной мощности LT2 (3 ммоль/л).

    ### ⚠️ Интерпретация алертов
    - **Критическое АД ≥250 мм рт.ст.** — немедленное прекращение теста (Löllgen).
    - **Чрезмерный подъём АД ≥220** — возможна латентная гипертензия.
    - **Хронотропная недостаточность** — пиковая ЧСС <85% от предсказанной, ассоциирована с плохим прогнозом.
    - **Восстановление ЧСС <12 уд/мин** — повышенный риск смертности (Löllgen).
    - **VO₂peak <25 мл/кг/мин** — высокий сердечно-сосудистый риск.

    Разработка: на основе открытых научных данных. Не является медицинским изделием.
    """)
