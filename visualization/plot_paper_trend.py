import os
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import const
from arguments import parse_args
from utility.utils_data import load_arXiv_data, load_tag2papers
from utility.utils_misc import project_setup


# penguins = sns.load_dataset("penguins")
# sns.ecdfplot(data=penguins, x="flipper_length_mm")
# plt.show()
# sns.ecdfplot(data=penguins, x="bill_length_mm", hue="species")
# plt.show()


def main():
    # Sample data
    data = {
        'year': [1991, 1992, 1993, 1994, 1995],
        'q-fin+econ': [1000, 1150, 1200, 1500, 1800],
        'q-bio': [500, 650, 700, 900, 1700],
        # Add other categories similarly
    }

    tag2papers = load_tag2papers(args)
    category2papers = {}

    path = osp.join(args.output_dir, "monthly_count.xlsx")

    if osp.exists(path):
        monthly_count = pd.read_excel(path)
        monthly_count['published'] = pd.to_datetime(monthly_count['published'], utc=True)
        monthly_count = monthly_count[monthly_count['published'] >= '1990-01']
        monthly_count.set_index('published', inplace=True)

    else:

        for category, subjects_li in const.ARXIV_SUBJECTS.items():
            category2papers[category] = set()
            for idx_subject, subject in enumerate(subjects_li):
                if subject not in tag2papers:
                    print(f"Subject {subject} not found")
                    continue

                print(f"[({category})\t{idx_subject}-th subject]\t{subject}\t{len(tag2papers[subject])} papers")
                category2papers[category].update(set(tag2papers[subject]))

        df = load_arXiv_data(args.data_dir)

        monthly_count_li = []

        for i, (category, ids_li) in enumerate(category2papers.items()):
            print(f"[{i}-th category]\t{category}\t{len(ids_li)} papers")
            papers_of_one_category = df[df[const.ID].isin(set(ids_li))]

            assert len(papers_of_one_category) == len(ids_li)

            monthly_count = papers_of_one_category.set_index('published').resample('MS').count().rename(columns={'id':
                                                                                                                     category})

            monthly_count_li += [monthly_count[category]]

        monthly_count = pd.concat(monthly_count_li, axis=1)

        monthly_count.index = monthly_count.index.strftime('%Y-%m')

        monthly_count.fillna(0).reset_index().to_excel(osp.join(args.output_dir, path), index=False)

    plt.figure(figsize=(12, 9))
    col1 = None
    col2 = None

    ax = monthly_count[list(const.ARXIV_SUBJECTS.keys())].plot(kind='area', stacked=True, alpha=0.5, colormap='viridis')



    # Add axis labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('#Papers')
    ax.set_title('#Monthly Submitted Papers by Subject')

    plt.tight_layout()
    plt.savefig(osp.join(args.output_dir, "monthly_count.pdf"), dpi=300)

    for i in range(len(const.ARXIV_SUBJECTS)):
        category1 = list(const.ARXIV_SUBJECTS.keys())[i]
        category2 = list(const.ARXIV_SUBJECTS.keys())[i + 1]

        melted = monthly_count.melt(id_vars='published', var_name='category', value_name='submissions')

        # sns.lineplot(data=df_melted, x='year', y='submissions', hue='category')

        col1 = df[category1] if col1 is None else col2
        col2 = col1 + df[category2]

        plt.fill_between(df['year'], col1, color=sns.color_palette()[0], alpha=0.5)
        plt.fill_between(df['year'], col1, col2, color=sns.color_palette()[1],
                         alpha=0.5)
    # Repeat the fill_between for other categories accordingly
    plt.title('New Submissions Over Year')
    plt.show()


    # sns.ecdfplot(data=df_melted, x='submissions', hue='category')
    # plt.title('ECDF of New Submissions')
    # plt.show()
    # col1 = col2


if __name__ == "__main__":
    project_setup()
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main()
