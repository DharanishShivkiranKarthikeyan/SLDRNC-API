// capi.cpp -- the public C interface: argument checks, exception -> status code, handle management.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>

#include "internal.hpp"

struct sldrnc_schema_s { sldi::Schema s; };
struct sldrnc_model_s { std::unique_ptr<sldi::Model> m; };
struct sldrnc_session_s { sldi::Session s; };
struct sldrnc_fleet_s { sldi::Fleet f; };

using namespace sldi;

static thread_local std::string g_err;
static sldrnc_status ok() { return SLDRNC_OK; }
static sldrnc_status err(sldrnc_status c, const std::string& m) { g_err = m; return c; }

#define SLD_TRY try {
#define SLD_CATCH                                                              \
    }                                                                          \
    catch (const sldi::Error& e) { return err(e.code, e.what()); }             \
    catch (const std::bad_alloc&) { return err(SLDRNC_ERROR_INTERNAL, "out of memory"); } \
    catch (const std::exception& e) { return err(SLDRNC_ERROR_INTERNAL, e.what()); }       \
    catch (...) { return err(SLDRNC_ERROR_INTERNAL, "unknown error"); }
#define NEED(cond, msg) do { if (!(cond)) return err(SLDRNC_ERROR_ARGUMENT, msg); } while (0)

extern "C" {

const char* sldrnc_last_error(void) { return g_err.c_str(); }
const char* sldrnc_version(void) { return "1.0.0"; }

// ------------------------------------------------------------------------------------------------ schema
sldrnc_status sldrnc_schema_create(double rate_hz, sldrnc_schema** out) {
    NEED(out, "out is NULL");
    NEED(rate_hz > 0 && std::isfinite(rate_hz), "rate_hz must be a positive number");
    SLD_TRY
    auto* s = new sldrnc_schema_s;
    s->s.rate = rate_hz;
    *out = s;
    return ok();
    SLD_CATCH
}
sldrnc_status sldrnc_schema_add_input(sldrnc_schema* schema, const char* name, sldrnc_input_kind kind, int size) {
    NEED(schema && name && *name, "schema or name is NULL / empty");
    NEED(kind >= SLDRNC_INPUT_POSITION && kind <= SLDRNC_INPUT_ANGLE, "unknown input kind");
    NEED(size > 0 && size <= 100000, "size must be positive");
    NEED(kind != SLDRNC_INPUT_QUATERNION || size == 4, "a QUATERNION channel has exactly 4 values (w, x, y, z)");
    SLD_TRY
    for (const auto& c : schema->s.ch) NEED(c.name != name, std::string("duplicate channel name '") + name + "'");
    Channel c;
    c.name = name; c.kind = kind; c.size = size; c.offset = schema->s.width;
    schema->s.ch.push_back(c);
    schema->s.width += size;
    return ok();
    SLD_CATCH
}
sldrnc_status sldrnc_schema_set_outputs(sldrnc_schema* schema, int n_outputs, const int* velocity_index) {
    NEED(schema, "schema is NULL");
    NEED(n_outputs > 0 && n_outputs <= 100000, "n_outputs must be positive");
    SLD_TRY
    schema->s.n_out = n_outputs;
    schema->s.vel_idx.assign(n_outputs, -1);
    if (velocity_index)
        for (int j = 0; j < n_outputs; j++) {
            NEED(velocity_index[j] >= -1 && velocity_index[j] < schema->s.width, "velocity_index[" + std::to_string(j) + "] is outside the input row (add the inputs first)");
            schema->s.vel_idx[j] = velocity_index[j];
        }
    return ok();
    SLD_CATCH
}
int sldrnc_schema_input_size(const sldrnc_schema* schema) { return schema ? schema->s.width : 0; }
int sldrnc_schema_output_size(const sldrnc_schema* schema) { return schema ? schema->s.n_out : 0; }
int sldrnc_schema_column(const sldrnc_schema* schema, const char* name) {
    if (!schema || !name) return -1;
    for (const auto& c : schema->s.ch) if (c.name == name) return c.offset;
    return -1;
}
sldrnc_status sldrnc_schema_set_history(sldrnc_schema* schema, int on) {
    NEED(schema, "schema is NULL");
    schema->s.history = on != 0;
    return ok();
}
sldrnc_status sldrnc_schema_set_affine(sldrnc_schema* schema, const char* name, int on) {
    NEED(schema && name, "schema or name is NULL");
    SLD_TRY
    Channel* c = nullptr;
    int others = 0;
    for (auto& ch : schema->s.ch) {
        if (ch.name == name) c = &ch;
        else if (!ch.affine) others++;
    }
    NEED(c, std::string("no channel named '") + name + "'");
    NEED(!on || c->kind != SLDRNC_INPUT_QUATERNION, "a QUATERNION channel cannot be affine");
    NEED(!on || others > 0, "at least one channel must stay non-affine");
    c->affine = on != 0;
    return ok();
    SLD_CATCH
}
void sldrnc_schema_free(sldrnc_schema* schema) { delete schema; }

// ------------------------------------------------------------------------------------------------ training
void sldrnc_train_options_default(sldrnc_train_options* o) {
    if (!o) return;
    o->size = SLDRNC_SIZE_MEDIUM;
    o->steps = 12000;
    o->batch = 512;
    o->learning_rate = 5e-3;
    o->weight_decay = 0.2;
    o->seed = 0;
    o->threads = 0;
    o->fit_correction = 1;
    o->verbose = 0;
    o->correction_delay_ticks = 1;
}
static void report_clear(sldrnc_report& r) {
    r.train_nmse = r.val_nmse = r.val_nmse_corrected = std::numeric_limits<double>::quiet_NaN();
    r.correction_time_ms = r.seconds = 0;
    r.macs_per_tick = 0;
    r.steps = 0;
    r.correction_delay_ticks = 0;
}
sldrnc_status sldrnc_train_full(const sldrnc_schema* schema, const double* X_train, const double* Y_train, size_t ticks_train,
                                const double* X_val, const double* Y_val, size_t ticks_val, const sldrnc_train_options* options,
                                sldrnc_model** out, sldrnc_report* report) {
    NEED(schema && X_train && Y_train && X_val && Y_val && out, "a required pointer is NULL");
    return sldrnc_train_full_multi(schema, 1, &X_train, &Y_train, &ticks_train, 1, &X_val, &Y_val, &ticks_val, options, out, report);
}
sldrnc_status sldrnc_train_full_multi(const sldrnc_schema* schema, size_t n_train, const double* const* X_train, const double* const* Y_train,
                                      const size_t* ticks_train, size_t n_val, const double* const* X_val, const double* const* Y_val,
                                      const size_t* ticks_val, const sldrnc_train_options* options, sldrnc_model** out,
                                      sldrnc_report* report) {
    NEED(schema && out, "schema or out is NULL");
    NEED(n_train > 0 && X_train && Y_train && ticks_train, "need at least one training recording (X_train, Y_train, ticks_train)");
    NEED(n_val > 0 && X_val && Y_val && ticks_val, "need at least one validation recording (X_val, Y_val, ticks_val)");
    SLD_TRY
    sldrnc_train_options o;
    sldrnc_train_options_default(&o);
    if (options) o = *options;
    NEED(o.size >= SLDRNC_SIZE_SMALL && o.size <= SLDRNC_SIZE_LARGE, "unknown size");
    NEED(o.correction_delay_ticks >= 1, "correction_delay_ticks must be at least 1 (1 = the next tick)");
    std::vector<Recording> tr, va;
    for (size_t k = 0; k < n_train; k++) tr.push_back({X_train[k], Y_train[k], ticks_train[k]});
    for (size_t k = 0; k < n_val; k++) va.push_back({X_val[k], Y_val[k], ticks_val[k]});
    sldrnc_report r;
    report_clear(r);
    auto m = train_full(schema->s, tr, va, o, r);
    auto* h = new sldrnc_model_s;
    h->m = std::move(m);
    *out = h;
    if (report) *report = r;
    return ok();
    SLD_CATCH
}

// ------------------------------------------------------------------------------------------------ RAW / CORRECTION
sldrnc_status sldrnc_model_from_layers(int n_layers, const sldrnc_layer* layers, const float* input_mean, const float* input_std,
                                       const float* output_mean, const float* output_std, sldrnc_model** out) {
    NEED(out && layers && n_layers > 0 && n_layers <= 64, "layers / out missing, or n_layers outside 1..64");
    SLD_TRY
    auto m = std::make_unique<Model>();
    m->mode = SLDRNC_MODE_RAW;
    m->net.L.resize(n_layers);
    for (int l = 0; l < n_layers; l++) {
        const sldrnc_layer& L = layers[l];
        NEED(L.in > 0 && L.out > 0 && L.W, "layer " + std::to_string(l) + ": sizes must be positive and W given");
        NEED(l == 0 || L.in == layers[l - 1].out, "layer " + std::to_string(l) + " expects " + std::to_string(L.in) + " inputs but the previous layer has " +
                                                   std::to_string(layers[l - 1].out) + " outputs");
        NEED(L.activation >= SLDRNC_ACT_IDENTITY && L.activation <= SLDRNC_ACT_SILU, "layer " + std::to_string(l) + ": unknown activation");
        for (size_t i = 0; i < (size_t)L.in * L.out; i++) NEED(std::isfinite(L.W[i]), "layer " + std::to_string(l) + ": W contains NaN / infinity");
        set_layer(m->net.L[l], L.in, L.out, L.activation, L.W, L.b);
    }
    const int nin = layers[0].in, nout = layers[n_layers - 1].out;
    NEED(!input_mean == !input_std, "give both input_mean and input_std, or neither");
    NEED(!output_mean == !output_std, "give both output_mean and output_std, or neither");
    if (input_mean) {
        m->in_mean.assign(input_mean, input_mean + nin);
        m->in_std.assign(input_std, input_std + nin);
        for (float v : m->in_std) NEED(v != 0 && std::isfinite(v), "input_std must be non-zero and finite");
    }
    if (output_mean) { m->out_mean.assign(output_mean, output_mean + nout); m->out_std.assign(output_std, output_std + nout); }
    auto* h = new sldrnc_model_s;
    h->m = std::move(m);
    *out = h;
    return ok();
    SLD_CATCH
}
static bool read_file(const char* path, std::vector<uint8_t>& b) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    if (n < 0) return false;
    b.resize((size_t)n);
    f.seekg(0);
    f.read((char*)b.data(), n);
    return (bool)f;
}
sldrnc_status sldrnc_model_import_onnx(const char* path, sldrnc_model** out) {
    NEED(path && out, "path or out is NULL");
    SLD_TRY
    std::vector<uint8_t> b;
    if (!read_file(path, b)) return err(SLDRNC_ERROR_IO, std::string("cannot read ") + path);
    auto* h = new sldrnc_model_s;
    try { h->m = import_onnx(b.data(), b.size()); } catch (...) { delete h; throw; }
    *out = h;
    return ok();
    SLD_CATCH
}
sldrnc_status sldrnc_model_import_onnx_memory(const void* data, size_t bytes, sldrnc_model** out) {
    NEED(data && bytes && out, "data / out is NULL or bytes is 0");
    SLD_TRY
    auto* h = new sldrnc_model_s;
    try { h->m = import_onnx((const uint8_t*)data, bytes); } catch (...) { delete h; throw; }
    *out = h;
    return ok();
    SLD_CATCH
}
sldrnc_status sldrnc_model_add_correction(sldrnc_model* model, const double* X, const double* Y, size_t ticks, double rate_hz,
                                          const int* velocity_index, int delay_ticks, sldrnc_report* report) {
    NEED(model && model->m && X && Y, "a required pointer is NULL");
    NEED(delay_ticks >= 1, "delay_ticks must be at least 1 (1 = the next tick)");
    SLD_TRY
    sldrnc_report r;
    report_clear(r);
    add_correction(*model->m, X, Y, ticks, rate_hz, velocity_index, delay_ticks, r);
    if (report) *report = r;
    return ok();
    SLD_CATCH
}
sldrnc_status sldrnc_model_tune_correction(const sldrnc_model* model, const double* X, const double* Y, size_t ticks, int delay_ticks,
                                           sldrnc_model** out, sldrnc_report* report) {
    NEED(model && model->m && X && Y && out, "a required pointer is NULL");
    NEED(delay_ticks >= 1, "delay_ticks must be at least 1 (1 = the next tick)");
    SLD_TRY
    sldrnc_report r;
    report_clear(r);
    auto m = tune_correction(*model->m, X, Y, ticks, delay_ticks, r);
    auto* h = new sldrnc_model_s;
    h->m = std::move(m);
    *out = h;
    if (report) *report = r;
    return ok();
    SLD_CATCH
}

sldrnc_status sldrnc_model_compile_table(const sldrnc_model* model, int n_grid, const int* grid, sldrnc_model** out) {
    NEED(model && model->m && out, "model or out is NULL");
    NEED(n_grid >= 0 && n_grid <= 3 && (n_grid == 0 || grid), "n_grid must be 0 (defaults) or 1..3 with grid given");
    SLD_TRY
    auto m = table_compile(*model->m, n_grid, grid);
    auto* h = new sldrnc_model_s;
    h->m = std::move(m);
    *out = h;
    return ok();
    SLD_CATCH
}

// ------------------------------------------------------------------------------------------------ models
sldrnc_status sldrnc_model_load(const char* path, sldrnc_model** out) {
    NEED(path && out, "path or out is NULL");
    SLD_TRY
    std::vector<uint8_t> b;
    if (!read_file(path, b)) return err(SLDRNC_ERROR_IO, std::string("cannot read ") + path);
    auto* h = new sldrnc_model_s;
    try { h->m = load_model(b.data(), b.size()); } catch (...) { delete h; throw; }
    *out = h;
    return ok();
    SLD_CATCH
}
sldrnc_status sldrnc_model_save(const sldrnc_model* model, const char* path) {
    NEED(model && model->m && path, "model or path is NULL");
    SLD_TRY
    std::vector<uint8_t> b;
    save_model(*model->m, b);
    std::ofstream f(path, std::ios::binary);
    if (!f) return err(SLDRNC_ERROR_IO, std::string("cannot write ") + path);
    f.write((const char*)b.data(), (std::streamsize)b.size());
    if (!f) return err(SLDRNC_ERROR_IO, std::string("cannot write ") + path);
    return ok();
    SLD_CATCH
}
void sldrnc_model_free(sldrnc_model* model) { delete model; }
sldrnc_status sldrnc_model_get_info(const sldrnc_model* model, sldrnc_model_info* info) {
    NEED(model && model->m && info, "model or info is NULL");
    const Model& m = *model->m;
    info->mode = (sldrnc_mode)m.mode;
    info->input_size = m.input_size();
    info->output_size = m.output_size();
    info->has_correction = m.corr.on ? 1 : 0;
    info->rate_hz = m.rate;
    info->correction_time_ms = m.corr.on && m.rate > 0 ? m.corr.tau_ticks * 1000.0 / m.rate : 0.0;
    info->macs_per_tick = m.macs();
    info->parameters = m.net.params();
    info->warmup_seconds = m.mode == SLDRNC_MODE_FULL ? warmup_seconds(m.schema) : 0.0;
    info->correction_delay_ticks = m.corr.on ? m.corr.delay_ticks : 0;
    info->history = m.mode == SLDRNC_MODE_FULL && m.schema.history ? 1 : 0;
    info->affine_inputs = m.n_aff;
    info->table_bytes = m.table.on() ? (int64_t)(m.table.T.size() * sizeof(float)) : 0;
    return ok();
}

// ------------------------------------------------------------------------------------------------ sessions
void sldrnc_session_options_default(sldrnc_session_options* o) {
    if (!o) return;
    o->use_correction = 1;
    o->protect = 1;
    o->max_delay_ticks = 0;
}
sldrnc_status sldrnc_session_create(const sldrnc_model* model, const sldrnc_session_options* options, sldrnc_session** out) {
    NEED(model && model->m && out, "model or out is NULL");
    SLD_TRY
    sldrnc_session_options o;
    sldrnc_session_options_default(&o);
    if (options) o = *options;
    NEED(o.max_delay_ticks >= 0 && o.max_delay_ticks <= 1000000, "max_delay_ticks must be between 0 (automatic) and 1000000");
    auto* s = new sldrnc_session_s;
    session_init(s->s, *model->m, o.use_correction != 0, o.protect != 0, o.max_delay_ticks);
    *out = s;
    return ok();
    SLD_CATCH
}
sldrnc_status sldrnc_session_step(sldrnc_session* session, const double* x, double* y) {
    if (!session || !x || !y) return err(SLDRNC_ERROR_ARGUMENT, "session, x or y is NULL");
    session_step(session->s, x, y);
    return SLDRNC_OK;
}
sldrnc_status sldrnc_session_observe(sldrnc_session* session, const double* y_measured) {
    if (!session || !y_measured) return err(SLDRNC_ERROR_ARGUMENT, "session or y_measured is NULL");
    session_observe(session->s, y_measured, 1);
    return SLDRNC_OK;
}
sldrnc_status sldrnc_session_observe_delayed(sldrnc_session* session, const double* y_measured, int delay_ticks) {
    if (!session || !y_measured) return err(SLDRNC_ERROR_ARGUMENT, "session or y_measured is NULL");
    if (delay_ticks < 1) return err(SLDRNC_ERROR_ARGUMENT, "delay_ticks must be at least 1 (1 = the tick just predicted)");
    if (session->s.ring_cap > 0 && delay_ticks > session->s.ring_cap)
        return err(SLDRNC_ERROR_ARGUMENT, "delay_ticks " + std::to_string(delay_ticks) + " exceeds this session's max_delay_ticks (" +
                                              std::to_string(session->s.ring_cap) + "); set max_delay_ticks when creating the session");
    session_observe(session->s, y_measured, delay_ticks);
    return SLDRNC_OK;
}
sldrnc_status sldrnc_session_reset(sldrnc_session* session) {
    NEED(session, "session is NULL");
    SLD_TRY
    session_init(session->s, *session->s.m, session->s.use_corr, session->s.protect, session->s.max_delay_opt);   // same options, fresh state
    return ok();
    SLD_CATCH
}
void sldrnc_session_free(sldrnc_session* session) { delete session; }

// ------------------------------------------------------------------------------------------------ fleets
void sldrnc_fleet_options_default(sldrnc_fleet_options* o) {
    if (!o) return;
    o->use_correction = 1;
    o->protect = 1;
    o->max_delay_ticks = 0;
    o->threads = 0;
}
sldrnc_status sldrnc_fleet_create(const sldrnc_model* model, size_t robots, const sldrnc_fleet_options* options, sldrnc_fleet** out) {
    NEED(model && model->m && out, "model or out is NULL");
    NEED(robots > 0 && robots <= (size_t)100000000, "robots must be between 1 and 100,000,000");
    SLD_TRY
    sldrnc_fleet_options o;
    sldrnc_fleet_options_default(&o);
    if (options) o = *options;
    NEED(o.max_delay_ticks >= 0 && o.max_delay_ticks <= 1000000, "max_delay_ticks must be between 0 (automatic) and 1000000");
    NEED(o.threads >= 0, "threads must be 0 (all cores) or more");
    auto* f = new sldrnc_fleet_s;
    try { fleet_init(f->f, *model->m, robots, o.use_correction != 0, o.protect != 0, o.max_delay_ticks, o.threads); } catch (...) { delete f; throw; }
    *out = f;
    return ok();
    SLD_CATCH
}
sldrnc_status sldrnc_fleet_step(sldrnc_fleet* fleet, const double* X, double* Y) {
    if (!fleet || !X || !Y) return err(SLDRNC_ERROR_ARGUMENT, "fleet, X or Y is NULL");
    fleet_step(fleet->f, X, Y);
    return SLDRNC_OK;
}
sldrnc_status sldrnc_fleet_observe(sldrnc_fleet* fleet, const double* Y_measured) {
    if (!fleet || !Y_measured) return err(SLDRNC_ERROR_ARGUMENT, "fleet or Y_measured is NULL");
    fleet_observe(fleet->f, Y_measured, 1);
    return SLDRNC_OK;
}
sldrnc_status sldrnc_fleet_observe_delayed(sldrnc_fleet* fleet, const double* Y_measured, int delay_ticks) {
    if (!fleet || !Y_measured) return err(SLDRNC_ERROR_ARGUMENT, "fleet or Y_measured is NULL");
    if (delay_ticks < 1) return err(SLDRNC_ERROR_ARGUMENT, "delay_ticks must be at least 1 (1 = the tick just predicted)");
    if (!fleet->f.rob.empty() && fleet->f.rob[0].ring_cap > 0 && delay_ticks > fleet->f.rob[0].ring_cap)
        return err(SLDRNC_ERROR_ARGUMENT, "delay_ticks " + std::to_string(delay_ticks) + " exceeds this fleet's max_delay_ticks (" +
                                              std::to_string(fleet->f.rob[0].ring_cap) + "); set max_delay_ticks when creating the fleet");
    fleet_observe(fleet->f, Y_measured, delay_ticks);
    return SLDRNC_OK;
}
sldrnc_status sldrnc_fleet_reset(sldrnc_fleet* fleet) {
    NEED(fleet, "fleet is NULL");
    SLD_TRY
    Fleet& f = fleet->f;
    fleet_init(f, *f.m, f.n, f.use_corr, f.protect, f.max_delay_opt, f.threads);
    return ok();
    SLD_CATCH
}
size_t sldrnc_fleet_size(const sldrnc_fleet* fleet) { return fleet ? fleet->f.n : 0; }
void sldrnc_fleet_free(sldrnc_fleet* fleet) { delete fleet; }

// ------------------------------------------------------------------------------------------------ offline
sldrnc_status sldrnc_model_run(const sldrnc_model* model, const sldrnc_session_options* options, const double* X, const double* Y, size_t ticks,
                               int delay_ticks, size_t score_from, double* Y_pred, double* nmse, double* nmse_per_output) {
    NEED(model && model->m && X, "model or X is NULL");
    NEED(ticks > 0, "ticks is 0");
    NEED(delay_ticks >= 1, "delay_ticks must be at least 1");
    NEED(score_from < ticks, "score_from must be less than ticks");
    SLD_TRY
    const Model& m = *model->m;
    sldrnc_session_options o;
    sldrnc_session_options_default(&o);
    if (options) o = *options;
    Session s;
    session_init(s, m, o.use_correction != 0, o.protect != 0, std::max(delay_ticks, o.max_delay_ticks));
    const int nin = m.input_size(), no = m.output_size();
    const int D = delay_ticks;
    std::vector<double> yp((size_t)no), se(no, 0.0), sm(no, 0.0), sm2(no, 0.0);
    for (size_t t = 0; t < ticks; t++) {
        session_step(s, X + t * nin, yp.data());
        for (int j = 0; j < no; j++) if (!std::isfinite(yp[j])) return err(SLDRNC_ERROR_DATA, "prediction is not finite at tick " + std::to_string(t) + " (check the input data)");
        if (Y_pred) std::memcpy(Y_pred + t * no, yp.data(), sizeof(double) * no);
        if (Y && t + 1 >= (size_t)D) session_observe(s, Y + (t + 1 - D) * no, D);   // the measurement of tick t + 1 - D arrives now
        if (Y && t >= score_from)
            for (int j = 0; j < no; j++) {
                const double y = Y[t * no + j], e = yp[j] - y;
                se[j] += e * e; sm[j] += y; sm2[j] += y * y;
            }
    }
    if (Y) {
        const double n = (double)(ticks - score_from);
        double tot = 0;
        for (int j = 0; j < no; j++) {
            const double var = sm2[j] / n - (sm[j] / n) * (sm[j] / n);
            const double v = se[j] / n / std::max(var, 1e-300);
            if (nmse_per_output) nmse_per_output[j] = v;
            tot += v;
        }
        if (nmse) *nmse = tot / no;
    } else if (nmse) {
        *nmse = std::numeric_limits<double>::quiet_NaN();
    }
    return ok();
    SLD_CATCH
}

}  // extern "C"
