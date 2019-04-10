#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#define EM_RANGE (0.01)

struct model_t
{
    float *em, *w, *b;
    float *em_bi, *w_bi;
    int64_t em_dim, vocab_num, category_num;
};

struct dataset_t
{
    int64_t *text_indices, *text_lens, *text_categories;
    int64_t *start_pos;
    int64_t text_num;  // number of word-sequence (line)
};

void init_model(struct model_t *model, int64_t em_dim, int64_t vocab_num, int64_t category_num, int64_t is_init)
{
    model->em_dim = em_dim;
    model->vocab_num = vocab_num;
    model->category_num = category_num;

    model->em = (float *)malloc(em_dim * vocab_num * sizeof(float));  // look up table for original
    model->em_bi = (float *)malloc(em_dim * vocab_num * sizeof(float));  // look up table for bi
    model->w = (float *)malloc(em_dim * category_num * sizeof(float));  // FC weight original
    model->w_bi = (float *)malloc(em_dim * category_num * sizeof(float));  // FC weight bi
    model->b = (float *)malloc(category_num * sizeof(float));  // FC bias

    float *em = model->em;
    float *em_bi = model->em_bi;
    float *w = model->w;
    float *w_bi = model->w_bi;
    float *b = model->b;
    int64_t i;
    if (is_init)
    {
        srand(time(NULL));
        // uniform distribution for look up table in U[-EM_RANGE, EM_RANGE]
        for (i = 0; i < em_dim * vocab_num; i++)
        {
            em[i] = ((float)rand() / RAND_MAX) * 2. * EM_RANGE - EM_RANGE;
            em_bi[i] = ((float)rand() / RAND_MAX) * 2. * EM_RANGE - EM_RANGE;
        }
        // uniform distribution for weight in U[-stdv, stdv]
        float stdv = 1. / (float)sqrt((double)em_dim * 2);
        for (i = 0; i < em_dim * category_num; i++)
        {
            w[i] = (float)rand() / RAND_MAX * 2. * stdv - stdv;
            w_bi[i] = (float)rand() / RAND_MAX * 2. * stdv - stdv;
        }

        for (i = 0; i < category_num; i++)
            b[i] = (float)rand() / RAND_MAX * 2. * stdv - stdv;
    }
    else
    {
        for (i = 0; i < em_dim * vocab_num; i++)
        {
            em[i] = 0.;
            em_bi[i] = 0.;
        }

        for (i = 0; i < em_dim * category_num; i++)
        {
            w[i] = 0.;
            w_bi[i] = 0.;
        }

        for (i = 0; i < category_num; i++)
            b[i] = 0.;
    }
}
void free_model(struct model_t *model)
{
    free(model->em);
    free(model->em_bi);
    free(model->w);
    free(model->w_bi);
    free(model->b);
}

int preread(FILE *fp)
{
    int ch = fgetc(fp);
    if (ch == EOF)
        return ch;
    else
    {
        fseek(fp, -1, SEEK_CUR);  // set the stream -fp- to a new position
        return ch;
    }
}
void load_data(struct dataset_t *data, const char *path, int64_t max_voc)
{
    FILE *fp = NULL;
    fp = fopen(path, "r");
    if (fp == NULL)
    {
        perror("error");
        exit(EXIT_FAILURE);
    }
    int next_i, next_ch;
    int64_t /*number of line*/text_num = 0, ch_num = 0, ignore_text_num = 0;
    int64_t text_len = 0;  // length of each sequence
    ;
    int64_t cat, text_i;
    enum state_t
    {
        READ_CAT,  // category
        READ_INDEX
    } state = READ_CAT;
    while (1)
    {
        int is_break = 0;
        switch (state)
        {
        case READ_CAT:  // read category label
            if (fscanf(fp, "%ld,", &cat) > 0)
            {
                if (preread(fp) == '\n')  // empty line
                {
                    ignore_text_num++;
                    fgetc(fp);  // read '\n'
                }
                else
                    state = READ_INDEX;
            }
            else  // end of file
            {
                assert(feof(fp));  // expression == false, abort; feof() == non-zero, is end
                is_break = 1;
            }
            break;
        case READ_INDEX:  // read word index
            assert(fscanf(fp, "%ld", &text_i) > 0);
            if (text_i < max_voc)  // current word in the vocabulary
            {
                ch_num++;
                text_len++;
            }
            next_ch = fgetc(fp);  // read ' ' or '\n'
            if (next_ch == '\n')  // end of current word-sequence
            {
                if (text_len == 0)  // empty line 
                {
                    ignore_text_num++;
                }
                else
                {
                    text_num++;  // increase the number of word-sequence
                    text_len = 0;  // reset the length of word-sequence
                }
                state = READ_CAT;
            }
        }
        if (is_break)
            break;
    }
    printf("load data from %s\n", path);
    printf("#lines: %ld, #chs: %ld\n", text_num, ch_num);
    printf("#ignore lines: %ld\n", ignore_text_num);
    data->text_num = text_num;
    data->text_indices = (int64_t *)malloc(ch_num * sizeof(int64_t));
    data->text_lens = (int64_t *)malloc(text_num * sizeof(int64_t));  // length of each word-seqence
    data->text_categories = (int64_t *)malloc(text_num * sizeof(int64_t));
    data->start_pos = (int64_t *)malloc(text_num * sizeof(int64_t));

    text_len = 0;
    int64_t *text_indices = data->text_indices;
    int64_t *text_lens = data->text_lens;
    int64_t *text_categories = data->text_categories;
    int64_t *start_pos = data->start_pos;
    rewind(fp);  // set position of stream to the beginning
    while (1)
    {
        int is_break = 0;
        switch (state)
        {
        case READ_CAT:  // read category label
            if (fscanf(fp, "%ld,", &cat) > 0)
            {
                if (preread(fp) == '\n')  // empty line
                {
                    fgetc(fp);  // read '\n'
                }
                else
                    state = READ_INDEX;
            }
            else  // end of file
            {
                assert(feof(fp));  //expression == false -> abort; feof() == non-zero -> is end
                is_break = 1;
            }
            break;
        case READ_INDEX:  // read word index
            assert(fscanf(fp, "%ld", &text_i) > 0);
            if (text_i < max_voc)  // current word in the vocabulary
            {
                text_len++;
                *text_indices = text_i;
                text_indices++;
            }
            next_ch = fgetc(fp);  // read ' ' or '\n'
            if (next_ch == '\n')  // end of current word-sequence
            {
                state = READ_CAT;
                if (text_len > 0)
                {
                    *text_lens = text_len;
                    text_lens++;
                    text_len = 0;  // reset the length of word-sequence

                    *text_categories = cat;  // set category label
                    text_categories++;
                }
            }
        }
        if (is_break)
            break;
    }
    start_pos[0] = 0;  // number of line
    for (int64_t i = 1; i < text_num; i++)
        start_pos[i] = start_pos[i - 1] + data->text_lens[i - 1];  // current pos = previous pos + previous length
    fclose(fp);
}

void free_data(struct dataset_t *data)
{
    free(data->text_indices);
    free(data->text_lens);
    free(data->text_categories);
    free(data->start_pos);
}

float forward(struct model_t *model, struct dataset_t *train_data, int64_t text_i, float *max_fea, int64_t *max_fea_index, float *max_bi_fea, int64_t *max_bi_fea_index, float *softmax_fea)
{  // load the text_i th word-sequence
    int64_t *text_indices = &(train_data->text_indices[train_data->start_pos[text_i]]);
    int64_t text_len = train_data->text_lens[text_i];
    assert(text_len >= 1);  // expression == true -> pass
    int64_t text_category = train_data->text_categories[text_i];

    int64_t i, j;
    int64_t em_pos, em_pos0, em_pos1;

    // max_pool original
    em_pos = text_indices[0] * model->em_dim;
    for (i = 0; i < model->em_dim; i++)
    {
        max_fea[i] = model->em[em_pos + i];
        max_fea_index[i] = em_pos + i;
    }

    for (i = 1; i < text_len; i++)
    {
        em_pos = text_indices[i] * model->em_dim;
        for (j = 0; j < model->em_dim; j++)
        {
            max_fea[j] = max_fea[j] > (model->em[em_pos + j]) ? max_fea[j] : (model->em[em_pos + j]);
            max_fea_index[j] = max_fea[j] > (model->em[em_pos + j]) ? max_fea_index[j] : (em_pos + j);
        }
    }

    // max_pool bi
    em_pos0 = text_indices[0] * model->em_dim;
    em_pos1 = (text_len > 1) ? (text_indices[1] * model->em_dim) : (text_indices[0] * model->em_dim); //长度为1 那么就把那个单词复制一个
    for (j = 0; j < model->em_dim; j++)
    {
        max_bi_fea[j] = (model->em_bi[em_pos0 + j] + model->em_bi[em_pos1 + j]) * 0.5;  // take average
        max_bi_fea_index[0] = em_pos0 + j;
        max_bi_fea_index[1] = em_pos1 + j;
    }

    if (text_len == 1)
    {
        // printf("warning: text[id: %ld] length == 1 (bi-gram features need length>1)\n", text_i);
    }

    for (i = 1; i < text_len - 1; i++)
    {
        em_pos0 = text_indices[i] * model->em_dim;
        em_pos1 = text_indices[i + 1] * model->em_dim;

        for (j = 0; j < model->em_dim; j++)
        {
            float fea = (model->em_bi[em_pos0 + j] + model->em_bi[em_pos1 + j]) * 0.5;  // take average
            if (max_bi_fea[j] < fea)
            {
                max_bi_fea[j] = fea;
                max_bi_fea_index[2 * j] = em_pos0 + j;
                max_bi_fea_index[2 * j + 1] = em_pos1 + j;
            }
        }
    }

    // mlp
    for (i = 0; i < model->category_num; i++)
        softmax_fea[i] = model->b[i];

    for (i = 0; i < model->category_num; i++)
        for (j = 0; j < model->em_dim; j++)
            softmax_fea[i] += (max_fea[j] * model->w[i * model->em_dim + j] + max_bi_fea[j] * model->w_bi[i * model->em_dim + j]);

    float loss = 0.;
    float tmp = 0.;
    loss -= softmax_fea[text_category];  // = log(exp(softmax))
    for (i = 0; i < model->category_num; i++)
    {
        softmax_fea[i] = (float)exp((double)softmax_fea[i]);
        tmp += softmax_fea[i];
    }
    loss += (float)log(tmp);  // loss = -log(exp(softmax/sigma(exp))
                               //      = log(sigma(exp)/exp(softmax)
                               //      = log(sigma(exp)) - softmax
    return loss;
}

void backward(struct model_t *model, struct dataset_t *train_data, int64_t text_i, float *max_fea, float *max_fea_bi, float *softmax_fea, float *grad_em, float *grad_em_bi, float *grad_w, float *grad_w_bi, float *grad_b)
{  // load the text_i th word-sequence
    int64_t *text_indices = &(train_data->text_indices[train_data->start_pos[text_i]]);
    int64_t text_len = train_data->text_lens[text_i];
    int64_t text_category = train_data->text_categories[text_i];

    float tmp_sum = 0.;
    int64_t i, j;
    for (i = 0; i < model->category_num; i++)
        tmp_sum += softmax_fea[i];
    for (i = 0; i < model->category_num; i++)
        grad_b[i] = softmax_fea[i] / tmp_sum;
    grad_b[text_category] -= 1.;  // error[text_category] = pre_softmax - 1
                                  // error[other] = pre_softmax - 0
                                  // error = predicted - labelled

    // weight original
    for (i = 0; i < model->category_num; i++)
        for (j = 0; j < model->em_dim; j++)
            grad_w[i * model->em_dim + j] = max_fea[j] * grad_b[i];

    // weight bi
    for (i = 0; i < model->category_num; i++)
        for (j = 0; j < model->em_dim; j++)
            grad_w_bi[i * model->em_dim + j] = max_fea_bi[j] * grad_b[i];

    // look up table original
    for (j = 0; j < model->em_dim; j++)
        grad_em[j] = 0.;
    for (i = 0; i < model->category_num; i++)
        for (j = 0; j < model->em_dim; j++)
            grad_em[j] += (model->w[i * model->em_dim + j]) * grad_b[i];

    // look up table bi
    for (j = 0; j < model->em_dim; j++)
        grad_em_bi[j] = 0.;
    for (i = 0; i < model->category_num; i++)
        for (j = 0; j < model->em_dim; j++)
            grad_em_bi[j] += (model->w_bi[i * model->em_dim + j]) * grad_b[i];
}

void evaluate(struct model_t *model, struct dataset_t *vali_data, int64_t batch_size, int64_t threads_n)
{
    printf("evaluating...\n");

    time_t eva_start, eva_end;
    eva_start = time(NULL);

    float *max_feas = (float *)malloc(model->em_dim * batch_size * sizeof(float));  // max feature original
    int64_t *max_fea_indexs = (int64_t *)malloc(model->em_dim * batch_size * sizeof(int64_t));
    float *max_bi_feas = (float *)malloc(model->em_dim * batch_size * sizeof(float));  // max feature bi
    int64_t *max_bi_fea_indexs = (int64_t *)malloc(2 * model->em_dim * batch_size * sizeof(int64_t));
    float *softmax_feas = (float *)malloc(model->category_num * batch_size * sizeof(float));  // input of softmax

    int64_t *pre_labels = (int64_t *)malloc(batch_size * sizeof(int64_t));  // predicted label
    int64_t *real_labels = (int64_t *)malloc(batch_size * sizeof(int64_t));
    
    float *cat_all = (float *)malloc(model->category_num * sizeof(float));
    float *cat_true = (float *)malloc(model->category_num * sizeof(float));

    for (int64_t i = 0; i < model->category_num; i++)
    {
        cat_all[i] = 0.;
        cat_true[i] = 0.;
    }

    for (int64_t batch_i = 0; batch_i < (vali_data->text_num + batch_size - 1) / batch_size; batch_i++)
    {// batch_i: index of batch
        int64_t real_batch_size = (vali_data->text_num - batch_i * batch_size) > batch_size ? batch_size : (vali_data->text_num - batch_i * batch_size);
        // 可以加速
#pragma omp parallel for schedule(dynamic) num_threads(threads_n)
        for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
        {// batch_j: index inside each batch
            int64_t text_i = (batch_i)*batch_size + batch_j;  // index of line
            assert(text_i < vali_data->text_num);  // expression == true -> pass

            int64_t text_category = vali_data->text_categories[text_i];
            // 长度为0的text，不计算梯度
            // 会导致问题，比如梯度没有更新
            // 应该在生成数据时避免
            if (vali_data->text_lens[text_i] == 0)
            {
                printf("error: vali text length can not be zero.[text id: %ld]", text_i);
                exit(-1);
            }

            float *max_fea = &max_feas[batch_j * model->em_dim];
            int64_t *max_fea_index = &max_fea_indexs[batch_j * model->em_dim];
            float *max_bi_fea = &max_bi_feas[batch_j * model->em_dim];
            int64_t *max_bi_fea_index = &max_bi_fea_indexs[2 * batch_j * model->em_dim];
            float *softmax_fea = &softmax_feas[batch_j * model->category_num];

            int64_t *pre_label = &pre_labels[batch_j];
            int64_t *real_label = &real_labels[batch_j];

            *real_label = text_category;

            forward(model, vali_data, text_i, max_fea, max_fea_index, max_bi_fea, max_bi_fea_index, softmax_fea);
            *pre_label = 0;
            float fea = softmax_fea[0];
            for (int64_t c = 1; c < model->category_num; c++)
            {// find max posibility (predicted category)
                if (softmax_fea[c] > fea)
                {
                    *pre_label = c;
                    fea = softmax_fea[c];
                }
            }
        }

        for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
        {// do statistics
            cat_all[real_labels[batch_j]] += 1;
            if (real_labels[batch_j] == pre_labels[batch_j])
                cat_true[real_labels[batch_j]] += 1;
        }
    }
    float cat_all_sum = 0.;
    float cat_true_sum = 0.;
    for (int64_t k = 0; k < model->category_num; k++)
    {
        cat_all_sum += cat_all[k];
        cat_true_sum += cat_true[k];
    }

    printf("#samples: %.0f\n", cat_all_sum);
    FILE *fp = fopen("fntext_bi_10_500.txt", "a");
    printf("macro precision: %.5f\n", cat_true_sum / cat_all_sum);
    fprintf(fp, "macro precision: %.5f\n", cat_true_sum / cat_all_sum);
    for (int64_t k = 0; k < model->category_num; k++)
    {
        printf("   category #%ld precision: %.5f\n", k, cat_true[k] / cat_all[k]);
        fprintf(fp, "   category #%ld precision: %.5f\n", k, cat_true[k] / cat_all[k]);
    }
    fclose(fp);

    free(max_feas);
    free(max_fea_indexs);
    free(max_bi_feas);
    free(max_bi_fea_indexs);
    free(softmax_feas);

    free(pre_labels);
    free(real_labels);
    free(cat_all);
    free(cat_true);

    eva_end = time(NULL);
    printf("   evaluating time: %lds\n", eva_end - eva_start);
}

void train_adam(struct model_t *model, struct dataset_t *train_data, struct dataset_t *vali_data, int64_t epochs, int64_t batch_size, int64_t threads_n)
{
    printf("start training(Adam)...\n");
    //     omp_lock_t omplock;
    // omp_init_lock(&omplock);

    int64_t tmp, i, sel;

    float alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    float beta1t = beta1;
    float beta2t = beta2;

    int64_t *shuffle_index = (int64_t *)malloc(train_data->text_num * sizeof(int64_t));  // number of line

    struct model_t adam_m, adam_v, gt;
    init_model(&adam_m, model->em_dim, model->vocab_num, model->category_num, 0);
    init_model(&adam_v, model->em_dim, model->vocab_num, model->category_num, 0);
    init_model(&gt, model->em_dim, model->vocab_num, model->category_num, 0);

    float *grads_em = (float *)malloc(model->em_dim * batch_size * sizeof(float));
    float *grads_em_bi = (float *)malloc(model->em_dim * batch_size * sizeof(float));
    float *grads_w = (float *)malloc(model->em_dim * model->category_num * batch_size * sizeof(float));
    float *grads_w_bi = (float *)malloc(model->em_dim * model->category_num * batch_size * sizeof(float));
    float *grads_b = (float *)malloc(model->category_num * batch_size * sizeof(float));

    float *max_feas = (float *)malloc(model->em_dim * batch_size * sizeof(float));
    int64_t *max_fea_indexs = (int64_t *)malloc(model->em_dim * batch_size * sizeof(int64_t));
    float *max_bi_feas = (float *)malloc(model->em_dim * batch_size * sizeof(float));
    int64_t *max_bi_fea_indexs = (int64_t *)malloc(2 * model->em_dim * batch_size * sizeof(int64_t));
    float *softmax_feas = (float *)malloc(model->category_num * batch_size * sizeof(float));
    float *losses = (float *)malloc(batch_size * sizeof(float));

    printf("init grad end...\n");

    for (i = 0; i < train_data->text_num; i++)
        shuffle_index[i] = i;

    for (int64_t epoch = 0; epoch < epochs; epoch++)
    {
        printf("#epoch: %ld\n", epoch);
        float s_loss = 0.;
        time_t epoch_start, epoch_end;
        // clock_t epoch_start, epoch_end;
        // shuffle
        for (i = 0; i < train_data->text_num; i++)
        {
            sel = rand() % (train_data->text_num - i) + i;  // random int in [i, text_num)
            tmp = shuffle_index[i];
            shuffle_index[i] = shuffle_index[sel];
            shuffle_index[sel] = tmp;
        }

        epoch_start = time(NULL);
        // epoch_start = clock();
        for (int64_t batch_i = 0; batch_i < (train_data->text_num + batch_size - 1) / batch_size; batch_i++)
        {
            int64_t real_batch_size = (train_data->text_num - batch_i * batch_size) > batch_size ? batch_size : (train_data->text_num - batch_i * batch_size);
            // 可以加速
#pragma omp parallel for schedule(dynamic) num_threads(threads_n)
            for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
            {
                int64_t text_i = (batch_i)*batch_size + batch_j;
                assert(text_i < train_data->text_num);  // expression = true -> pass
                text_i = shuffle_index[text_i];

                // 长度为0的text，不计算梯度
                // 会导致问题，比如梯度没有更新
                // 应该在生成数据时避免
                if (train_data->text_lens[text_i] == 0)
                {
                    printf("error: training text length can not be zero.[text id: %ld]", text_i);
                    exit(-1);
                }

                float *grad_em = &grads_em[batch_j * model->em_dim];
                float *grad_em_bi = &grads_em_bi[batch_j * model->em_dim];
                float *grad_w = &grads_w[batch_j * model->em_dim * model->category_num];
                float *grad_w_bi = &grads_w_bi[batch_j * model->em_dim * model->category_num];
                float *grad_b = &grads_b[batch_j * model->category_num];

                float *max_fea = &max_feas[batch_j * model->em_dim];
                int64_t *max_fea_index = &max_fea_indexs[batch_j * model->em_dim];
                float *max_bi_fea = &max_bi_feas[batch_j * model->em_dim];
                int64_t *max_bi_fea_index = &max_bi_fea_indexs[2 * batch_j * model->em_dim];
                float *softmax_fea = &softmax_feas[batch_j * model->category_num];

                losses[batch_j] = forward(model, train_data, text_i, max_fea, max_fea_index, max_bi_fea, max_bi_fea_index, softmax_fea);
                backward(model, train_data, text_i, max_fea, max_bi_fea, softmax_fea, grad_em, grad_em_bi, grad_w, grad_w_bi, grad_b);
            }

            for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
                s_loss += losses[batch_j];

            // adding grad of multiple batch
            for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
            {
                // weight original
                for (int64_t batch_k = 0; batch_k < model->em_dim * model->category_num; batch_k++)
                    gt.w[batch_k] += grads_w[batch_j * model->em_dim * model->category_num + batch_k] / (float)batch_size;
                // weight bi
                for (int64_t batch_k = 0; batch_k < model->em_dim * model->category_num; batch_k++)
                    gt.w_bi[batch_k] += grads_w_bi[batch_j * model->em_dim * model->category_num + batch_k] / (float)batch_size;
                // bias
                for (int64_t batch_k = 0; batch_k < model->category_num; batch_k++)
                    gt.b[batch_k] += grads_b[batch_j * model->category_num + batch_k] / (float)batch_size;
                
                // embedding table
                for (int64_t batch_k = 0; batch_k < model->em_dim; batch_k++)
                {
                    // look up table original
                    int64_t em_index = max_fea_indexs[batch_j * model->em_dim + batch_k];
                    gt.em[em_index] += grads_em[batch_j * model->em_dim + batch_k] / (float)batch_size;

                    // look up table bi
                    int64_t em_index0 = max_bi_fea_indexs[2 * batch_j * model->em_dim + 2 * batch_k];
                    int64_t em_index1 = max_bi_fea_indexs[2 * batch_j * model->em_dim + 2 * batch_k + 1];
                    gt.em_bi[em_index0] += 0.5 * grads_em_bi[batch_j * model->em_dim + batch_k] / (float)batch_size;  // take average
                    gt.em_bi[em_index1] += 0.5 * grads_em_bi[batch_j * model->em_dim + batch_k] / (float)batch_size;  // take average
                }
            }

                // 计算m,v update param 可以加速
#pragma omp parallel for schedule(static) num_threads(threads_n)
            for (int64_t batch_k = 0; batch_k < model->em_dim * model->category_num; batch_k++)
            {
                // weight original
                adam_m.w[batch_k] = beta1 * adam_m.w[batch_k] + (1 - beta1) * gt.w[batch_k];
                adam_v.w[batch_k] = beta2 * adam_v.w[batch_k] + (1 - beta2) * gt.w[batch_k] * gt.w[batch_k];
                gt.w[batch_k] = 0.;

                float m_hat = adam_m.w[batch_k] / (1 - beta1t);
                float v_hat = adam_v.w[batch_k] / (1 - beta2t);
                model->w[batch_k] -= alpha * m_hat / ((float)sqrt((float)v_hat) + epsilon);

                // weight bi
                adam_m.w_bi[batch_k] = beta1 * adam_m.w_bi[batch_k] + (1 - beta1) * gt.w_bi[batch_k];
                adam_v.w_bi[batch_k] = beta2 * adam_v.w_bi[batch_k] + (1 - beta2) * gt.w_bi[batch_k] * gt.w_bi[batch_k];
                gt.w_bi[batch_k] = 0.;

                m_hat = adam_m.w_bi[batch_k] / (1 - beta1t);
                v_hat = adam_v.w_bi[batch_k] / (1 - beta2t);
                model->w_bi[batch_k] -= alpha * m_hat / ((float)sqrt((float)v_hat) + epsilon);
            }

            // 循环数量少，不用加速
            // bias
            for (int64_t batch_k = 0; batch_k < model->category_num; batch_k++)
            {
                adam_m.b[batch_k] = beta1 * adam_m.b[batch_k] + (1 - beta1) * gt.b[batch_k];
                adam_v.b[batch_k] = beta2 * adam_v.b[batch_k] + (1 - beta2) * gt.b[batch_k] * gt.b[batch_k];
                gt.b[batch_k] = 0.;

                float m_hat = adam_m.b[batch_k] / (1 - beta1t);
                float v_hat = adam_v.b[batch_k] / (1 - beta2t);
                model->b[batch_k] -= alpha * m_hat / ((float)sqrt((float)v_hat) + epsilon);
            }

            for (int64_t batch_j = 0; batch_j < real_batch_size; batch_j++)
            {
                // embedding table
                for (int64_t batch_k = 0; batch_k < model->em_dim; batch_k++)
                {
                    // look up table original
                    int64_t em_index = max_fea_indexs[batch_j * model->em_dim + batch_k];
                    if (gt.em[em_index] != 0.)
                    {
                        adam_m.em[em_index] = beta1 * adam_m.em[em_index] + (1 - beta1) * gt.em[em_index];
                        adam_v.em[em_index] = beta2 * adam_v.em[em_index] + (1 - beta2) * gt.em[em_index] * gt.em[em_index];
                        gt.em[em_index] = 0.;

                        float m_hat = adam_m.em[em_index] / (1 - beta1t);
                        float v_hat = adam_v.em[em_index] / (1 - beta2t);
                        model->em[em_index] -= alpha * m_hat / ((float)sqrt((float)v_hat) + epsilon);
                    }

                    // look up table bi
                    int64_t em_index0 = max_bi_fea_indexs[2 * batch_j * model->em_dim + 2 * batch_k];
                    int64_t em_index1 = max_bi_fea_indexs[2 * batch_j * model->em_dim + 2 * batch_k + 1];

                    if (gt.em_bi[em_index0] != 0.)
                    {
                        adam_m.em_bi[em_index0] = beta1 * adam_m.em_bi[em_index0] + (1 - beta1) * gt.em_bi[em_index0];
                        adam_v.em_bi[em_index0] = beta2 * adam_v.em_bi[em_index0] + (1 - beta2) * gt.em_bi[em_index0] * gt.em_bi[em_index0];
                        gt.em_bi[em_index0] = 0.;

                        float m_hat = adam_m.em_bi[em_index0] / (1 - beta1t);
                        float v_hat = adam_v.em_bi[em_index0] / (1 - beta2t);
                        model->em_bi[em_index0] -= alpha * m_hat / ((float)sqrt((float)v_hat) + epsilon);
                    }
                    if (gt.em_bi[em_index1] != 0.)
                    {
                        adam_m.em_bi[em_index1] = beta1 * adam_m.em_bi[em_index1] + (1 - beta1) * gt.em_bi[em_index1];
                        adam_v.em_bi[em_index1] = beta2 * adam_v.em_bi[em_index1] + (1 - beta2) * gt.em_bi[em_index1] * gt.em_bi[em_index1];
                        gt.em_bi[em_index1] = 0.;

                        float m_hat = adam_m.em_bi[em_index1] / (1 - beta1t);
                        float v_hat = adam_v.em_bi[em_index1] / (1 - beta2t);
                        model->em_bi[em_index1] -= alpha * m_hat / ((float)sqrt((float)v_hat) + epsilon);
                    }
                }
            }

            beta1t *= beta1t;
            beta2t *= beta2t;

        } // end_batch
        epoch_end = time(NULL);
        // epoch_end = clock();

        s_loss /= train_data->text_num;
        printf("    loss: %.4f\n", s_loss);
        printf("    time: %lds\n", epoch_end - epoch_start);
        // printf("    time: %.1fs\n", (double)(epoch_end - epoch_start)/CLOCKS_PER_SEC );

        if (vali_data != NULL)
        {
            printf("evaluate vali data...\n");
            evaluate(model, vali_data, batch_size, threads_n);
        }

        printf("\n");

    } //end_epoch
    free(shuffle_index);
    free_model(&adam_m);
    free_model(&adam_v);
    free_model(&gt);
    free(grads_em);
    free(grads_em_bi);
    free(grads_w);
    free(grads_w_bi);
    free(grads_b);
    free(max_feas);
    free(max_fea_indexs);
    free(max_bi_feas);
    free(max_bi_fea_indexs);
    free(softmax_feas);
    free(losses);
}
void show(int64_t *a, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
        printf("%ld ", a[i]);
    printf("\n");
}

int arg_helper(char *str, int argc, char **argv)
{
    int pos;
    for (pos = 1; pos < argc; pos++)
        if (strcmp(str, argv[pos]) == 0)
            return pos;
    return -1;
}

void save_em(struct model_t *model, char *path, int64_t n)
{
    FILE *fp = NULL;
    fp = fopen(path, "w");
    if (fp == NULL)
    {
        perror("error");
        exit(EXIT_FAILURE);
    }
    for (int64_t i = 0; i < n; i++)
    {
        int64_t pos = i * model->em_dim;
        for (int64_t j = 0; j < model->em_dim; j++)
        {
            if (j == model->em_dim - 1)
            {
                fprintf(fp, "%.8f\n", model->em[pos + j]);
            }
            else
            {
                fprintf(fp, "%.8f ", model->em[pos + j]);
            }
        }
    }
    fclose(fp);
}
int main(int argc, char **argv)
{
    struct model_t model;
    struct dataset_t train_data, vali_data, test_data;

    int64_t em_dim = 200, vocab_num = 0, category_num = 0, em_len = 0;
    int64_t epochs = 10, batch_size = 2000, threads_n = 20;
    float lr = 0.5, limit_vocab=1.;
    char *train_data_path = NULL, *vali_data_path = NULL, *test_data_path = NULL, *em_path = NULL;

    int i;
    if ((i = arg_helper("-dim", argc, argv)) > 0)
        em_dim = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-vocab", argc, argv)) > 0)
        vocab_num = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-category", argc, argv)) > 0)
        category_num = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-epoch", argc, argv)) > 0)
        epochs = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-batch-size", argc, argv)) > 0)
        batch_size = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-thread", argc, argv)) > 0)
        threads_n = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-lr", argc, argv)) > 0)
        lr = (float)atof(argv[i + 1]);
    if ((i = arg_helper("-train", argc, argv)) > 0)
        train_data_path = argv[i + 1];
    if ((i = arg_helper("-vali", argc, argv)) > 0)
        vali_data_path = argv[i + 1];
    if ((i = arg_helper("-test", argc, argv)) > 0)
        test_data_path = argv[i + 1];
    if ((i = arg_helper("-em-path", argc, argv)) > 0)
        em_path = argv[i + 1];
    if ((i = arg_helper("-em-len", argc, argv)) > 0)
        em_len = (int64_t)atoi(argv[i + 1]);
    if ((i = arg_helper("-limit-vocab", argc, argv)) > 0)
        limit_vocab = (float)atof(argv[i + 1]);

    if (vocab_num == 0)
    {
        printf("error: miss -vocab");
        exit(-1);
    }
    if (category_num == 0)
    {
        printf("error: miss -category");
        exit(-1);
    }
    if (train_data_path == NULL)
    {
        printf("error: need train data!");
        exit(-1);
    }

    init_model(&model, em_dim, vocab_num, category_num, 1);

    if (train_data_path != NULL)
        load_data(&train_data, train_data_path, (int64_t)(limit_vocab*vocab_num));
    if (test_data_path != NULL)
        load_data(&test_data, test_data_path, (int64_t)(limit_vocab*vocab_num));
    if (vali_data_path != NULL)
        load_data(&vali_data, vali_data_path, (int64_t)(limit_vocab*vocab_num));

    if (vali_data_path != NULL)
        train_adam(&model, &train_data, &vali_data, epochs, batch_size, threads_n);
    else
        train_adam(&model, &train_data, NULL, epochs, batch_size, threads_n);

    if (test_data_path != NULL)
    {
        printf("evaluate test data...\n");
        evaluate(&model, &test_data, batch_size, threads_n);
    }

    if (em_path != NULL)
    {
        printf("saving em...\n");
        if (em_len == 0)
            em_len = model.vocab_num;
        save_em(&model, em_path, em_len);
    }

    free_model(&model);
    if (train_data_path != NULL)
        free_data(&train_data);
    if (test_data_path != NULL)
        free_data(&test_data);
    if (vali_data_path != NULL)
        free_data(&vali_data);

    return 0;
}
