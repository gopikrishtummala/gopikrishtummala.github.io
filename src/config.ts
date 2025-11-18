export const SITE = {
  website: "https://gopikrishtummala.github.io/", // replace this with your deployed domain
  author: "Gopi Krishna Tummala",
  profile: "https://gopikrishtummala.github.io/",
  desc: "Senior Machine Learning Engineer @ Adobe | PhD | Ex-Qualcomm | Ex-Zoox | IIT-Madras",
  title: "Gopi Krishna Tummala",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 6,
  postPerPage: 6,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "https://github.com/gopikrishtummala/gopikrishtummala.github.io/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "America/Los_Angeles", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
